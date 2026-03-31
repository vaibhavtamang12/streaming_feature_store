from confluent_kafka.admin import AdminClient, NewTopic
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

TOPICS = [
    NewTopic(
        settings.kafka_transactions_topic,
        num_partitions=settings.kafka_partitions,
        replication_factor=settings.kafka_replication_factor,
        config={
            "retention.ms": str(7 * 24 * 60 * 60 * 1000),  # 7 days
            "compression.type": "lz4",
            "min.insync.replicas": "1",
        },
    ),
    NewTopic(
        settings.kafka_user_activity_topic,
        num_partitions=settings.kafka_partitions,
        replication_factor=settings.kafka_replication_factor,
    ),
]

def provision_topics():
    admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap_servers})
    futures = admin.create_topics(TOPICS, validate_only=False)
    for topic, future in futures.items():
        try:
            future.result()
            logger.info(f"Topic '{topic}' created.")
        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Topic '{topic}' already exists, skipping.")
            else:
                raise