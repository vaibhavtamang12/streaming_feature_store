from feast import Entity, FeatureView, Field
from feast.types import Float32, Int64
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)

# Entity
user = Entity(name="user_id", join_keys=["user_id"])

# Data source (Postgres table)
postgres_source = PostgreSQLSource(
    name="user_features_source",
    query="""
        SELECT user_id,
               txn_count_1min,
               txn_sum_1min,
               event_timestamp
        FROM user_features
    """,
    timestamp_field="event_timestamp",
)

# Feature View
user_features_view = FeatureView(
    name="user_features",
    entities=[user],
    ttl=None,
    schema=[
        Field(name="txn_count_1min", dtype=Int64),
        Field(name="txn_sum_1min", dtype=Float32),
    ],
    online=True,
    source=postgres_source,
)