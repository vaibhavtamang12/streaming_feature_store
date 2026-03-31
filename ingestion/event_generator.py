import asyncio
import json
import random
import time
import uuid
from datetime import datetime
from confluent_kafka import Producer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
from confluent_kafka.serialization import SerializationContext, MessageField
from config.settings import settings

# Realistic merchant distribution
MERCHANT_CATEGORIES = ["grocery", "restaurant", "gas_station", "e-commerce",
                        "travel", "entertainment", "health", "atm_withdrawal"]
HIGH_RISK_CATEGORIES = {"atm_withdrawal", "e-commerce"}

class TransactionGenerator:
    def __init__(self, target_tps: int = 150):
        self.target_tps = target_tps
        self.producer = Producer({
            "bootstrap.servers": "localhost:9092",
            "linger.ms": 5,            # Micro-batching for throughput
            "batch.size": 65536,
            "compression.type": "lz4",
            "acks": "1",               # Leader ack only for low latency
        })
        self._user_pool = [str(uuid.uuid4()) for _ in range(500)]
        self._merchant_pool = [str(uuid.uuid4()) for _ in range(200)]

    def _generate_transaction(self) -> dict:
        is_fraud = random.random() < 0.02   # 2% fraud rate
        category = random.choice(MERCHANT_CATEGORIES)
        amount = (
            random.uniform(500, 5000)       # Fraud: unusual amount
            if is_fraud
            else random.lognormvariate(3.5, 1.2)  # Normal: log-normal spend
        )
        return {
            "transaction_id": str(uuid.uuid4()),
            "user_id": random.choice(self._user_pool),
            "amount": round(abs(amount), 2),
            "merchant_id": random.choice(self._merchant_pool),
            "merchant_category": category,
            "currency": "USD",
            "timestamp_ms": int(time.time() * 1000),
            "ip_address": f"192.168.{random.randint(0,255)}.{random.randint(0,255)}",
            "device_id": str(uuid.uuid4()),
            "is_fraud": is_fraud,
        }

    def _delivery_report(self, err, msg):
        if err:
            print(f"Delivery failed: {err}")

    async def run(self):
        interval = 1.0 / self.target_tps
        print(f"Generating {self.target_tps} transactions/sec...")
        while True:
            tx = self._generate_transaction()
            self.producer.produce(
                settings.kafka_transactions_topic,
                key=tx["user_id"],           # Partition by user for ordering
                value=json.dumps(tx).encode(),
                on_delivery=self._delivery_report,
            )
            self.producer.poll(0)            # Non-blocking poll
            await asyncio.sleep(interval)

if __name__ == "__main__":
    gen = TransactionGenerator(target_tps=150)
    asyncio.run(gen.run())