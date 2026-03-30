import json
import time
from collections import deque
from kafka import KafkaConsumer
from db import insert_features
import redis

KAFKA_BROKER = "redpanda:9092"
TOPIC = "transactions"

REDIS_HOST = "redis"
REDIS_PORT = 6379

# Redis client
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Kafka consumer
consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="feature-group"
)

# In-memory store for sliding window (per user)
user_windows = {}

WINDOW_SIZE = 60  # seconds

def update_features(user_id, amount, timestamp):
    if user_id not in user_windows:
        user_windows[user_id] = deque()

    window = user_windows[user_id]

    # Add new transaction
    window.append((timestamp, amount))

    # Remove old transactions
    while window and (timestamp - window[0][0]) > WINDOW_SIZE:
        window.popleft()

    # Compute features
    txn_count = len(window)
    txn_sum = sum(a for _, a in window)

    return txn_count, txn_sum


def store_features(user_id, txn_count, txn_sum):
    key = f"user_features:{user_id}"

    r.hset(key, mapping={
        "txn_count_1min": txn_count,
        "txn_sum_1min": txn_sum
    })


def main():
    print("Starting stream processor...")

    for message in consumer:
        txn = message.value

        user_id = txn["user_id"]
        amount = txn["amount"]
        timestamp = txn["timestamp"]

        txn_count, txn_sum = update_features(user_id, amount, timestamp)

        store_features(user_id, txn_count, txn_sum)

        insert_features(user_id, txn_count, txn_sum, timestamp)
        
        print(f"User {user_id} → count={txn_count}, sum={txn_sum}")



if __name__ == "__main__":
    main()