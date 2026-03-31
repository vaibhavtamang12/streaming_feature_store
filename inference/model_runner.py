import numpy as np
import onnxruntime as ort
import mlflow
from functools import lru_cache
from config.settings import settings

FEATURE_ORDER = [
    "tx_count_10m", "tx_count_1h", "tx_amount_avg_1h", "tx_amount_sum_1h",
    "tx_amount_max_1h", "distinct_merchants_1h", "high_risk_category_flag",
    "velocity_score",
]

@lru_cache(maxsize=1)
def get_session() -> ort.InferenceSession:
    """Load ONNX model once, cache in memory. Thread-safe via lru_cache."""
    client = mlflow.MlflowClient(settings.mlflow_tracking_uri)
    latest = client.get_latest_versions(settings.model_name, stages=["Production"])[0]
    local_path = mlflow.artifacts.download_artifacts(latest.source)
    return ort.InferenceSession(
        local_path,
        providers=["CPUExecutionProvider"],   # swap to CUDAExecutionProvider for GPU
    )

def predict(features: dict) -> dict:
    session = get_session()
    input_name = session.get_inputs()[0].name

    vector = np.array(
        [[float(features.get(f, 0.0)) for f in FEATURE_ORDER]],
        dtype=np.float32,
    )
    outputs = session.run(None, {input_name: vector})
    fraud_prob = float(outputs[1][0][1])   # probability of class=1 (fraud)

    return {
        "fraud_probability": round(fraud_prob, 4),
        "is_fraud": fraud_prob >= 0.5,
        "confidence": round(max(fraud_prob, 1 - fraud_prob), 4),
        "model_version": get_session()._model_meta.version,
    }