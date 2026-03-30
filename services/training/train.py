import pandas as pd
import psycopg2
import mlflow
import numpy as np
import os
import mlflow.sklearn
import time
from config import DB_CONFIG, TRAINING_WINDOW_MINUTES, MLFLOW_TRACKING_URI
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from skl2onnx import to_onnx

DB_CONFIG = {
    "host": "postgres",
    "database": "feature_store",
    "user": "postgres",
    "password": "postgres"
}

MLFLOW_TRACKING_URI = "http://mlflow:5000"


def load_data():
    conn = psycopg2.connect(**DB_CONFIG)

    current_time = int(time.time())
    window_start = current_time - (TRAINING_WINDOW_MINUTES * 60)

    query = f"""
    SELECT user_id, txn_count_1min, txn_sum_1min
    FROM user_features
    WHERE event_timestamp >= {window_start}
    """

    df = pd.read_sql(query, conn)
    conn.close()

    return df


def create_labels(df):
    """
    Fake labels for now (simulate fraud)
    """
    df["label"] = (
        (df["txn_count_1min"] > 5) |
        (df["txn_sum_1min"] > 3000)
    ).astype(int)

    return df


def train():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("fraud-detection")

    df = load_data()
    df = create_labels(df)

    X = df[["txn_count_1min", "txn_sum_1min"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    with mlflow.start_run():

        model = LogisticRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Convert to ONNX
        sample_input = np.array([[1, 100.0]], dtype=np.float32)

        onnx_model = to_onnx(model, sample_input)

# Save ONNX model
        os.makedirs("artifacts", exist_ok=True)

        with open("artifacts/model.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())

        print("ONNX model saved!")

        mlflow.log_param("model", "logistic_regression")
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(model, "model")

        print(f"Model Accuracy: {acc}")




if __name__ == "__main__":
    train()