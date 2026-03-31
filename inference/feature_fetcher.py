"""
Feature fetcher — abstracts all Feast/Redis interaction from the API layer.
Swap this module to change your online store without touching api.py.
"""
import logging
from typing import Optional
import redis.asyncio as aioredis
from feast import FeatureStore
from config.settings import settings

logger = logging.getLogger(__name__)

# Default feature values for cold-start users (no history in the store yet)
COLD_START_DEFAULTS = {
    "tx_count_10m":             0,
    "tx_count_1h":              0,
    "tx_amount_avg_1h":         0.0,
    "tx_amount_sum_1h":         0.0,
    "tx_amount_max_1h":         0.0,
    "distinct_merchants_1h":    0,
    "high_risk_category_flag":  0,
    "velocity_score":           0.0,
}

FEAST_FEATURES = [
    "transaction_stats:tx_count_10m",
    "transaction_stats:tx_count_1h",
    "transaction_stats:tx_amount_avg_1h",
    "transaction_stats:tx_amount_sum_1h",
    "transaction_stats:tx_amount_max_1h",
    "transaction_stats:distinct_merchants_1h",
    "transaction_stats:high_risk_category_flag",
    "transaction_stats:velocity_score",
]


class FeatureFetcher:
    """
    Wraps Feast's online store retrieval.
    Designed to be instantiated once at app startup (via FastAPI lifespan)
    and reused across all requests — Feast maintains its own Redis pool.
    """

    def __init__(self, feast_store: FeatureStore):
        self._store = feast_store

    def get_features(
        self,
        user_id: str,
        merchant_category: Optional[str] = None,
    ) -> dict:
        """
        Fetch features for a single user from the Feast online store (Redis).

        Returns a flat dict ready for the ONNX model input vector.
        Falls back to cold-start defaults if the user has no stored history.

        Args:
            user_id:           The entity key to look up in Feast.
            merchant_category: Passed in from the live request — not stored
                               in the feature store, computed inline.

        Returns:
            Feature dict with all keys in COLD_START_DEFAULTS guaranteed present.
        """
        try:
            raw = self._store.get_online_features(
                features=FEAST_FEATURES,
                entity_rows=[{"user_id": user_id}],
            ).to_dict()

            # Feast returns lists for batch compatibility — unwrap the first element
            features = {
                key.split(":")[-1]: (values[0] if isinstance(values, list) else values)
                for key, values in raw.items()
                if key != "user_id"
            }

            # Merge with defaults: any None values (feature exists but no data) → default
            resolved = {
                k: (features.get(k) if features.get(k) is not None else v)
                for k, v in COLD_START_DEFAULTS.items()
            }

        except Exception as e:
            logger.warning(
                f"Feast lookup failed for user '{user_id}', "
                f"using cold-start defaults. Error: {e}"
            )
            resolved = dict(COLD_START_DEFAULTS)

        # Inject the live request-time feature (not stored, computed per call)
        resolved["high_risk_category_flag"] = int(
            merchant_category in {"atm_withdrawal", "e-commerce", "wire_transfer"}
        ) if merchant_category else 0

        logger.debug(f"Features for {user_id}: {resolved}")
        return resolved

    def get_features_batch(self, user_ids: list[str]) -> list[dict]:
        """
        Batch fetch for multiple users in a single Redis round-trip.
        Useful for bulk scoring endpoints or shadow-mode evaluation.
        """
        try:
            raw = self._store.get_online_features(
                features=FEAST_FEATURES,
                entity_rows=[{"user_id": uid} for uid in user_ids],
            ).to_dict()

            n = len(user_ids)
            results = []
            for i in range(n):
                row = {
                    k.split(":")[-1]: (
                        raw[k][i] if raw[k][i] is not None
                        else COLD_START_DEFAULTS.get(k.split(":")[-1], 0.0)
                    )
                    for k in FEAST_FEATURES
                }
                results.append(row)
            return results

        except Exception as e:
            logger.error(f"Batch feature fetch failed: {e}")
            return [dict(COLD_START_DEFAULTS) for _ in user_ids]