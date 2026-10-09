CREATE TABLE IF NOT EXISTS {database}.lens_call_costs
(
    key_kind LowCardinality(String),
    key_value String,
    team_id LowCardinality(String),
    start_time DateTime64(3),
    request_id String,
    litellm_call_id String,
    response_id String,
    upstream_response_id String,
    provider_request_id String,
    trace_id String,
    span_id String,
    api_key String,
    user String,
    spend Nullable(Float64) DEFAULT NULL,
    end_time DateTime64(3),
    EngineReceivedMs UInt64 DEFAULT 0
)
ENGINE = ReplacingMergeTree(end_time)
PARTITION BY toYYYYMM(start_time)
ORDER BY (key_kind, key_value, team_id, start_time, request_id)
SETTINGS materialize_ttl_recalculate_only = 1, non_replicated_deduplication_window = 1000
