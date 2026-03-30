from fastapi import FastAPI
from routes.features import router as feature_router
from routes.predict import router as predict_router
from onnx_model import load_model
from feast_client import init_store
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import time
import os

app = FastAPI()

@app.on_event("startup")
def startup_event():
    load_model()
    init_store()

@app.get("/")
def root():
    return {"message": "Streaming Feature Store API Running 🚀"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "kafka": os.getenv("KAFKA_BROKER"),
        "redis": os.getenv("REDIS_HOST"),
        "mlflow": os.getenv("MLFLOW_TRACKING_URI")
    }

REQUEST_COUNT = Counter(
    "api_requests_total", "Total API Requests", ["method", "endpoint"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "Request latency", ["endpoint"]
)

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()

    response = await call_next(request)

    latency = time.time() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()

    REQUEST_LATENCY.labels(
        endpoint=request.url.path
    ).observe(latency)

    return response

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

app.include_router(feature_router)
app.include_router(predict_router)