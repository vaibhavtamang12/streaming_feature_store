"""
Offline pipeline. Generate training data.
Quick start: python -m training.offline_pipeline --synthetic
Real data:   python -m training.offline_pipeline
"""
import argparse
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from feast import FeatureStore

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR       = Path("data")
ENTITY_DF_PATH   = OUTPUT_DIR / "entity_df.parquet"
TRAINING_DF_PATH = OUTPUT_DIR / "training_features.parquet"

FEATURE_COLUMNS = [
    "tx_count_10m", "tx_count_1h", "tx_amount_avg_1h",
    "tx_amount_sum_1h", "tx_amount_max_1h", "distinct_merchants_1h",
    "high_risk_category_flag", "velocity_score",
]

FEAST_FEATURES = [f"transaction_stats:{c}" for c in FEATURE_COLUMNS]


def generate_synthetic(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    rng      = np.random.default_rng(seed)
    is_fraud = rng.random(n) < 0.02

    df = pd.DataFrame({
        "is_fraud":                is_fraud.astype(int),
        "tx_count_10m":            np.where(is_fraud, rng.integers(5, 20, n),   rng.integers(0, 3, n)),
        "tx_count_1h":             np.where(is_fraud, rng.integers(10, 40, n),  rng.integers(1, 7, n)),
        "tx_amount_avg_1h":        np.where(is_fraud, rng.uniform(400, 2000, n), np.abs(rng.lognormal(3.5, 1, n))),
        "tx_amount_sum_1h":        np.where(is_fraud, rng.uniform(1000, 8000, n), np.abs(rng.lognormal(5, 1.2, n))),
        "tx_amount_max_1h":        np.where(is_fraud, rng.uniform(500, 5000, n), np.abs(rng.lognormal(4, 1, n))),
        "distinct_merchants_1h":   np.where(is_fraud, rng.integers(4, 10, n),   rng.integers(1, 3, n)),
        "high_risk_category_flag": rng.integers(0, 2, n),
        "velocity_score":          np.where(is_fraud, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n)),
    })
    logger.info(f"Synthetic: {len(df)} rows | fraud={df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
    return df


def generate_entity_df(n_users: int = 500, n_events: int = 50_000) -> pd.DataFrame:
    rng    = np.random.default_rng(42)
    now    = datetime.now(tz=timezone.utc)
    users  = [str(uuid.uuid4()) for _ in range(n_users)]

    df = pd.DataFrame({
        "user_id":         rng.choice(users, size=n_events),
        "event_timestamp": [
            now - timedelta(seconds=int(s))
            for s in rng.integers(0, 90 * 24 * 3600, size=n_events)
        ],
        "is_fraud": rng.random(n_events) < 0.02,
    })
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
    return df.sort_values("event_timestamp").reset_index(drop=True)


def join_historical(entity_df: pd.DataFrame) -> pd.DataFrame:
    store = FeatureStore(repo_path=settings.feast_repo_path)
    logger.info("Fetching historical features (point-in-time join)...")
    df = store.get_historical_features(
        entity_df=entity_df,
        features=FEAST_FEATURES,
    ).to_df()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0)
    df["is_fraud"] = df["is_fraud"].astype(int)
    return df


def run(use_synthetic: bool = False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if use_synthetic:
        df = generate_synthetic()
    else:
        entity_df = generate_entity_df()
        entity_df.to_parquet(ENTITY_DF_PATH, index=False)
        df = join_historical(entity_df)

    df.to_parquet(TRAINING_DF_PATH, index=False)
    logger.info(f"Saved training data → {TRAINING_DF_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    run(use_synthetic=args.synthetic)