CREATE TABLE IF NOT EXISTS {database}.spans_core
(
    TeamId           LowCardinality(String),
    ApiKeyHash       String,
    UserId           String,
    TraceId          String,
    SpanId           String,
    ParentSpanId     String,
    TraceRef         String DEFAULT hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))),
    Timestamp        DateTime64(9) CODEC(Delta, ZSTD(1)),
    Duration         UInt64 CODEC(ZSTD(1)),
    EngineReceivedMs UInt64 DEFAULT 0,
    SpanName         LowCardinality(String),
    ServiceName      LowCardinality(String),
    StatusCode       LowCardinality(String),
    StatusMessage    String CODEC(ZSTD(1)),
    ObservationType  LowCardinality(String),
    AgentName        LowCardinality(String),
    Framework        LowCardinality(String),
    Model            LowCardinality(String),
    InputTokens      UInt32,
    OutputTokens     UInt32,
    InputPreview     String CODEC(ZSTD(1)),
    INDEX idx_trace_id  TraceId  TYPE bloom_filter(0.001) GRANULARITY 1,
    INDEX idx_trace_ref TraceRef TYPE bloom_filter(0.001) GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toDate(Timestamp)
ORDER BY (TeamId, toStartOfHour(Timestamp), ApiKeyHash, TraceId, SpanId)
SETTINGS ttl_only_drop_parts = 1, materialize_ttl_recalculate_only = 1, non_replicated_deduplication_window = 1000
