"""
Stream processor — pure confluent-kafka consumer.
Replaces Bytewax. Runs as a long-lived process: one thread per partition.
"""
import json
import logging
import signal
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from confluent_kafka import Consumer, KafkaError, KafkaException
from feast import FeatureStore

from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Rolling window state ──────────────────────────────────────────────────────

class UserFeatureState:
    """Thread-safe per-user sliding window."""

    def __init__(self):
        self._lock = threading.Lock()
        self.window_10m: deque = deque()   # (timestamp_ms, amount)
        self.window_1h:  deque = deque()

    def add(self, amount: float, ts_ms: int):
        with self._lock:
            self.window_10m.append((ts_ms, amount))
            self.window_1h.append((ts_ms, amount))
            self._evict(ts_ms)

    def _evict(self, now_ms: int):
        cutoff_10m = now_ms - 10 * 60 * 1000
        cutoff_1h  = now_ms - 60 * 60 * 1000
        while self.window_10m and self.window_10m[0][0] < cutoff_10m:
            self.window_10m.popleft()
        while self.window_1h and self.window_1h[0][0] < cutoff_1h:
            self.window_1h.popleft()

    def compute(self) -> dict:
        with self._lock:
            amounts_1h = [a for _, a in self.window_1h]
            count_1h   = len(amounts_1h)
            return {
                "tx_count_10m":          len(self.window_10m),
                "tx_count_1h":           count_1h,
                "tx_amount_avg_1h":      sum(amounts_1h) / max(count_1h, 1),
                "tx_amount_sum_1h":      sum(amounts_1h),
                "tx_amount_max_1h":      max(amounts_1h, default=0.0),
                "distinct_merchants_1h": 0,        # extend: track merchant set
                "velocity_score":        min(len(self.window_10m) / 5.0, 1.0),
            }


# Shared across all consumer threads — protected by UserFeatureState's own lock
_user_states: dict[str, UserFeatureState] = defaultdict(UserFeatureState)
_states_lock = threading.Lock()


def get_state(user_id: str) -> UserFeatureState:
    with _states_lock:
        return _user_states[user_id]


# ── Feast writer ──────────────────────────────────────────────────────────────

class FeastWriter:
    """Batches feature rows and pushes to Feast online store (Redis)."""

    def __init__(self, feast_repo_path: str, batch_size: int = 50):
        self._store      = FeatureStore(repo_path=feast_repo_path)
        self._batch_size = batch_size
        self._buffer: list[dict] = []
        self._lock = threading.Lock()

    def write(self, features: dict):
        with self._lock:
            self._buffer.append(features)
            if len(self._buffer) >= self._batch_size:
                self._flush()

    def flush(self):
        with self._lock:
            self._flush()

    def _flush(self):
        if not self._buffer:
            return
        try:
            df = pd.DataFrame(self._buffer)
            df["event_timestamp"] = pd.to_datetime(
                df["event_timestamp"], utc=True
            )
            self._store.push(
                push_source_name="transaction_stats_push",
                df=df,
                to="online",
            )
            logger.debug(f"Flushed {len(self._buffer)} rows to Feast/Redis")
        except Exception as e:
            logger.error(f"Feast push failed: {e}")
        finally:
            self._buffer.clear()


# ── Consumer worker ───────────────────────────────────────────────────────────

class ConsumerWorker(threading.Thread):
    """One Kafka consumer thread. Stateless per-thread; state lives in _user_states."""

    def __init__(self, worker_id: int, feast_writer: FeastWriter, stop_event: threading.Event):
        super().__init__(name=f"consumer-{worker_id}", daemon=True)
        self._writer     = feast_writer
        self._stop_event = stop_event
        self._consumer   = Consumer({
            "bootstrap.servers":  settings.kafka_bootstrap_servers,
            "group.id":           "stream-processor-group",
            "auto.offset.reset":  "latest",
            "enable.auto.commit": True,
            "max.poll.interval.ms": 300_000,
        })

    def run(self):
        self._consumer.subscribe([settings.kafka_transactions_topic])
        logger.info(f"{self.name} subscribed to {settings.kafka_transactions_topic}")

        flush_counter = 0
        try:
            while not self._stop_event.is_set():
                msg = self._consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    raise KafkaException(msg.error())

                self._process(msg)
                flush_counter += 1
                if flush_counter % 200 == 0:
                    self._writer.flush()

        except Exception as e:
            logger.error(f"{self.name} crashed: {e}", exc_info=True)
        finally:
            self._consumer.close()
            logger.info(f"{self.name} shut down cleanly")

    def _process(self, msg):
        try:
            tx = json.loads(msg.value().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Bad message: {e}")
            return

        user_id  = tx.get("user_id")
        amount   = float(tx.get("amount", 0.0))
        ts_ms    = int(tx.get("timestamp_ms", time.time() * 1000))
        category = tx.get("merchant_category", "")

        if not user_id:
            return

        state = get_state(user_id)
        state.add(amount, ts_ms)
        features = state.compute()

        features.update({
            "user_id":                 user_id,
            "event_timestamp":         datetime.now(tz=timezone.utc).isoformat(),
            "high_risk_category_flag": int(category in {"atm_withdrawal", "e-commerce", "wire_transfer"}),
        })

        self._writer.write(features)


# ── Entrypoint ────────────────────────────────────────────────────────────────

def run(num_workers: int = 4):
    stop_event   = threading.Event()
    feast_writer = FeastWriter(
        feast_repo_path=settings.feast_repo_path,
        batch_size=50,
    )

    workers = [
        ConsumerWorker(i, feast_writer, stop_event)
        for i in range(num_workers)
    ]

    def _shutdown(sig, frame):
        logger.info("Shutting down stream processor...")
        stop_event.set()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    for w in workers:
        w.start()

    logger.info(f"Stream processor running with {num_workers} workers.")
    stop_event.wait()

    feast_writer.flush()
    for w in workers:
        w.join(timeout=10)

    logger.info("Stream processor stopped.")


if __name__ == "__main__":
    run(num_workers=4)