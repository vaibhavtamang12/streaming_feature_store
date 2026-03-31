from datetime import timedelta
from feast import Entity, FeatureView, Field, PushSource, FileSource
from feast.types import Float64, Int64
from feast import ValueType   # 🔥 ADD THIS

user = Entity(
    name="user_id",
    value_type=ValueType.INT64,   # ✅ FIXED
    description="User performing the transaction",
)

transaction_push_source = PushSource(
    name="transaction_stats_push",
    batch_source=FileSource(
        path="../data/transaction_stats.parquet",
        timestamp_field="event_timestamp",
    ),
)

transaction_stats_view = FeatureView(
    name="transaction_stats",
    entities=[user],
    ttl=timedelta(hours=24),
    schema=[
        Field(name="user_id", dtype=Int64),
        Field(name="tx_count_10m", dtype=Int64),
        Field(name="tx_count_1h", dtype=Int64),
        Field(name="tx_amount_avg_1h", dtype=Float64),
        Field(name="tx_amount_sum_1h", dtype=Float64),
        Field(name="tx_amount_max_1h", dtype=Float64),
        Field(name="distinct_merchants_1h", dtype=Int64),
        Field(name="high_risk_category_flag", dtype=Int64),
        Field(name="velocity_score", dtype=Float64),
    ],
    source=transaction_push_source,
)