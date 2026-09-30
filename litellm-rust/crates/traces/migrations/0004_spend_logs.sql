CREATE TABLE IF NOT EXISTS {database}.spend_logs
(
    request_id            String,
    response_id           String,
    call_type             LowCardinality(String),
    api_key               String,
    key_alias             String,
    team_id               LowCardinality(String),
    team_alias            String,
    organization_id       String,
    user                  String,
    end_user              String,
    model                 LowCardinality(String),
    model_group           LowCardinality(String),
    model_id              String,
    custom_llm_provider   LowCardinality(String),
    api_base              String,
    spend                 Float64,
    prompt_tokens         UInt32,
    completion_tokens     UInt32,
    total_tokens          UInt32,
    cache_read_tokens     UInt32,
    cache_write_tokens    UInt32,
    start_time            DateTime64(3),
    end_time              DateTime64(3),
    completion_start_time Nullable(DateTime64(3)),
    status                LowCardinality(String),
    error_str             String,
    cache_hit             Bool,
    session_id            String,
    trace_id              String,
    span_id               String,
    request_tags          Array(String),
    metadata              String CODEC(ZSTD(3)),
    messages              String CODEC(ZSTD(3)),
    response              String CODEC(ZSTD(3)),
    INDEX idx_response_id response_id TYPE bloom_filter(0.001) GRANULARITY 1,
    INDEX idx_trace_id    trace_id    TYPE bloom_filter(0.001) GRANULARITY 1
)
ENGINE = ReplacingMergeTree(end_time)
PARTITION BY toYYYYMM(start_time)
ORDER BY (team_id, toDateTime(start_time), request_id)
TTL toDateTime(start_time) + INTERVAL {spend_log_retention_days} DAY
