import json
import time
import random
from kafka import KafkaProducer
from faker import Faker

fake = Faker()

KAFKA_BROKER = "redpanda:9092"
TOPIC = "transactions"

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

def generate_transaction():
    return {
        "transaction_id": fake.uuid4(),
        "user_id": random.randint(1, 100),
        "amount": round(random.uniform(10, 5000), 2),
        "merchant": fake.company(),
        "location": fake.city(),
        "timestamp": int(time.time()),
    }

if __name__ == "__main__":
    print("Starting transaction producer...")

    while True:
        txn = generate_transaction()
        producer.send(TOPIC, txn)
        print(f"Sent: {txn}")
        time.sleep(1)