from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Swap Kafka ↔ Redpanda by changing this one line
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_transactions_topic: str = "transactions"
    kafka_user_activity_topic: str = "user_activity"
    kafka_partitions: int = 12   # 12 allows scaling to 12 consumers
    kafka_replication_factor: int = 1

    # Swap Redis ↔ DynamoDB by implementing the OnlineStore protocol
    redis_url: str = "redis://localhost:6379"
    redis_feature_ttl_seconds: int = 3600

    feast_repo_path: str = "./feature_repo"
    mlflow_tracking_uri: str = "http://localhost:5000"
    model_name: str = "fraud_detector"

    class Config:
        env_file = ".env"

settings = Settings()