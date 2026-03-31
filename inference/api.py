"""
FastAPI inference gateway — full file.
Receives a transaction, fetches features from Feast/Redis,
runs ONNX inference, returns fraud score.
P95 target: <150ms
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from feast import FeatureStore

from config.settings import settings
from inference.feature_fetcher import FeatureFetcher
from inference.model_runner import predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────

REQUEST_LATENCY = Histogram(
    "fraud_api_request_duration_seconds",
    "End-to-end request latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0],
)
FEATURE_FETCH_LATENCY = Histogram(
    "feast_feature_fetch_duration_seconds",
    "Feast/Redis feature retrieval latency",
    buckets=[0.001, 0.002, 0.005, 0.01, 0.025, 0.05],
)
INFERENCE_LATENCY = Histogram(
    "onnx_inference_duration_seconds",
    "ONNX model inference latency",
    buckets=[0.001, 0.002, 0.005, 0.01, 0.025],
)
FRAUD_COUNTER = Counter(
    "fraud_detections_total",
    "Total number of transactions flagged as fraud",
)
LEGIT_COUNTER = Counter(
    "legit_transactions_total",
    "Total number of transactions scored as legitimate",
)
ERROR_COUNTER = Counter(
    "fraud_api_errors_total",
    "Total number of API errors",
    ["error_type"],
)

# ── App lifespan (startup / shutdown) ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup — initialises Feast store and FeatureFetcher,
    stores them on app.state so every request can reuse the same connection pool.
    """
    logger.info("Initialising Feast feature store...")
    try:
        feast_store = FeatureStore(repo_path=settings.feast_repo_path)
        app.state.fetcher = FeatureFetcher(feast_store)
        logger.info("Feature fetcher ready.")
    except Exception as e:
        logger.error(f"Failed to initialise Feast store: {e}")
        raise

    logger.info("Warming up ONNX model...")
    try:
        # Dummy call so the model is loaded into memory before the first real request
        from inference.model_runner import get_session
        get_session()
        logger.info("ONNX model loaded and ready.")
    except Exception as e:
        logger.warning(f"ONNX warm-up failed (will retry on first request): {e}")

    yield

    logger.info("Shutting down inference API.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection Inference API",
    description="Real-time fraud scoring via Feast features + ONNX model.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Request / Response schemas ────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    transaction_id: str
    user_id: str
    amount: float
    merchant_category: str | None = None

    model_config = {"json_schema_extra": {
        "example": {
            "transaction_id": "txn-abc-123",
            "user_id": "user-xyz-456",
            "amount": 2500.00,
            "merchant_category": "atm_withdrawal",
        }
    }}


class ScoreResponse(BaseModel):
    transaction_id: str
    user_id: str
    fraud_probability: float
    is_fraud: bool
    confidence: float
    model_version: str
    features_used: dict
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    feast_ok: bool
    model_ok: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/score",
    response_model=ScoreResponse,
    summary="Score a transaction for fraud",
)
async def score_transaction(request: ScoreRequest, req: Request):
    """
    Main scoring endpoint.

    Flow:
      1. Fetch real-time features from Feast (Redis online store)
      2. Run ONNX model inference
      3. Record Prometheus metrics
      4. Return fraud probability + decision
    """
    t0 = time.perf_counter()

    # ── Step 1: Feature retrieval ─────────────────────────────────────────
    t_feat = time.perf_counter()
    try:
        feat_dict = req.app.state.fetcher.get_features(
            user_id=request.user_id,
            merchant_category=request.merchant_category,
        )
    except Exception as e:
        ERROR_COUNTER.labels(error_type="feature_fetch").inc()
        logger.error(f"Feature fetch failed for user {request.user_id}: {e}")
        raise HTTPException(
            status_code=503,
            detail="Feature store temporarily unavailable. Please retry.",
        )
    FEATURE_FETCH_LATENCY.observe(time.perf_counter() - t_feat)

    # ── Step 2: ONNX inference ────────────────────────────────────────────
    t_infer = time.perf_counter()
    try:
        result = predict(feat_dict)
    except Exception as e:
        ERROR_COUNTER.labels(error_type="inference").inc()
        logger.error(f"Inference failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Model inference failed. Please retry.",
        )
    INFERENCE_LATENCY.observe(time.perf_counter() - t_infer)

    # ── Step 3: Metrics ───────────────────────────────────────────────────
    if result["is_fraud"]:
        FRAUD_COUNTER.inc()
        logger.info(
            f"FRAUD detected | tx={request.transaction_id} "
            f"user={request.user_id} prob={result['fraud_probability']:.4f}"
        )
    else:
        LEGIT_COUNTER.inc()

    total_ms = (time.perf_counter() - t0) * 1000
    REQUEST_LATENCY.observe(total_ms / 1000)

    if total_ms > 150:
        logger.warning(
            f"P95 SLA breach: {total_ms:.1f}ms "
            f"(tx={request.transaction_id})"
        )

    return ScoreResponse(
        transaction_id=request.transaction_id,
        user_id=request.user_id,
        fraud_probability=result["fraud_probability"],
        is_fraud=result["is_fraud"],
        confidence=result["confidence"],
        model_version=result.get("model_version", "unknown"),
        features_used=feat_dict,
        latency_ms=round(total_ms, 2),
    )


@app.post(
    "/score/batch",
    summary="Score multiple transactions in one call",
)
async def score_batch(requests: list[ScoreRequest], req: Request):
    """
    Batch scoring — useful for bulk evaluation or shadow mode.
    Runs each transaction sequentially (ONNX is fast enough that
    parallelism overhead exceeds benefit at small batch sizes).
    """
    if len(requests) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size limit is 100 transactions per call.",
        )

    results = []
    for r in requests:
        try:
            feat_dict = req.app.state.fetcher.get_features(
                user_id=r.user_id,
                merchant_category=r.merchant_category,
            )
            result = predict(feat_dict)
            results.append({
                "transaction_id":    r.transaction_id,
                "user_id":           r.user_id,
                "fraud_probability": result["fraud_probability"],
                "is_fraud":          result["is_fraud"],
                "confidence":        result["confidence"],
            })
            if result["is_fraud"]:
                FRAUD_COUNTER.inc()
            else:
                LEGIT_COUNTER.inc()
        except Exception as e:
            logger.error(f"Batch item failed (tx={r.transaction_id}): {e}")
            results.append({
                "transaction_id": r.transaction_id,
                "error":          str(e),
            })

    return {"results": results, "count": len(results)}


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
async def health(req: Request):
    """
    Checks both the Feast connection and the ONNX model.
    Returns 200 if fully healthy, 503 if any component is down.
    """
    feast_ok = False
    model_ok = False

    try:
        # Lightweight Feast ping — list feature views (no Redis call)
        req.app.state.fetcher._store.list_feature_views()
        feast_ok = True
    except Exception as e:
        logger.warning(f"Feast health check failed: {e}")

    try:
        from inference.model_runner import get_session
        get_session()
        model_ok = True
    except Exception as e:
        logger.warning(f"Model health check failed: {e}")

    status_code = 200 if (feast_ok and model_ok) else 503
    return Response(
        content=HealthResponse(
            status="ok" if (feast_ok and model_ok) else "degraded",
            feast_ok=feast_ok,
            model_ok=model_ok,
        ).model_dump_json(),
        status_code=status_code,
        media_type="application/json",
    )


@app.get(
    "/metrics",
    summary="Prometheus metrics scrape endpoint",
)
async def metrics():
    """Exposes all Prometheus counters and histograms for Grafana."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "Fraud Detection API",
        "docs":    "/docs",
        "health":  "/health",
        "metrics": "/metrics",
    }