"""
Training pipeline (fixed + production-ready)
Run: python -m training.train_model
"""

import logging
from pathlib import Path

import mlflow
import mlflow.lightgbm
import mlflow.onnx
import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score

import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

from config.settings import settings

# ─────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "tx_count_10m",
    "tx_count_1h",
    "tx_amount_avg_1h",
    "tx_amount_sum_1h",
    "tx_amount_max_1h",
    "distinct_merchants_1h",
    "high_risk_category_flag",
    "velocity_score",
]

TRAINING_DF_PATH = Path("data/training_features.parquet")
MODEL_DIR = Path("models")

# ─────────────────────────────────────────────────────────────


def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAINING_DF_PATH.exists():
        raise FileNotFoundError(
            f"{TRAINING_DF_PATH} not found. "
            "Run: python -m training.offline_pipeline --synthetic"
        )

    # ── Load Data ─────────────────────────────────────────────
    df = pd.read_parquet(TRAINING_DF_PATH)

    X = df[FEATURE_COLUMNS].fillna(0).astype(np.float32)
    y = df["is_fraud"].astype(int)

    logger.info(f"Dataset: {len(df)} rows | fraud rate: {y.mean()*100:.2f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── Handle Imbalance ──────────────────────────────────────
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": pos_weight,
        "n_estimators": 300,
        "verbosity": -1,
    }

    # ── MLflow Setup ─────────────────────────────────────────
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("fraud_detection")

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        mlflow.log_params(params)

        # ── Train Model ──────────────────────────────────────
        model = lgb.LGBMClassifier(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[
                lgb.early_stopping(30, verbose=False),
                lgb.log_evaluation(50),
            ],
        )

        # ── Predictions (FIXED) ──────────────────────────────
        preds_prob = model.predict_proba(X_test)[:, 1]

        # 🔥 Threshold tuning (important for fraud detection)
        threshold = 0.3
        preds = (preds_prob > threshold).astype(int)

        # ── Metrics ──────────────────────────────────────────
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, preds_prob)
        f1 = f1_score(y_test, preds, zero_division=0)

        metrics = {
            "precision": prec,
            "recall": rec,
            "roc_auc": auc,
            "f1": f1,
            "threshold": threshold,
        }

        mlflow.log_metrics(metrics)
        logger.info(f"Metrics: {metrics}")

        # ── Save Native Model (optional but recommended) ─────
        mlflow.lightgbm.log_model(model, "lgb_model")

        # ── ONNX Conversion ─────────────────────────────────
        initial_types = [
            ("float_input", FloatTensorType([None, len(FEATURE_COLUMNS)]))
        ]

        onnx_model = onnxmltools.convert_lightgbm(
            model,
            initial_types=initial_types,
            target_opset=12,
        )

        # Save locally
        onnx_path = MODEL_DIR / "fraud_detector.onnx"
        onnxmltools.utils.save_model(onnx_model, str(onnx_path))

        logger.info(f"ONNX model saved to {onnx_path}")

        # ── Log ONNX Model (FIXED) ──────────────────────────
        mlflow.onnx.log_model(
            onnx_model,
            artifact_path="fraud_detector"
        )

        # Log dataset (optional)
        mlflow.log_artifact(str(TRAINING_DF_PATH))

        # ── Register Model (FIXED) ──────────────────────────
        model_uri = f"runs:/{run_id}/fraud_detector"

        mv = mlflow.register_model(model_uri, settings.model_name)

        logger.info(
            f"Registered model v{mv.version} | Run ID: {run_id}"
        )

        return run_id


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()