CREATE TABLE IF NOT EXISTS {database}.lens_feedback
(
    TeamId           LowCardinality(String),
    ApiKeyHash       String,
    TraceId          String CODEC(ZSTD(1)),
    Author           String,
    Score            UInt8,
    Comment          String CODEC(ZSTD(3)),
    CreatedAt        DateTime64(3),
    UpdatedAt        DateTime64(3),
    IsDeleted        UInt8,
    EngineReceivedMs UInt64 DEFAULT 0,
    INDEX idx_trace_id TraceId TYPE bloom_filter(0.001) GRANULARITY 1,
    CONSTRAINT score_range CHECK Score <= 10
)
ENGINE = ReplacingMergeTree(UpdatedAt, IsDeleted)
ORDER BY (TeamId, ApiKeyHash, TraceId, Author)
SETTINGS materialize_ttl_recalculate_only = 1, non_replicated_deduplication_window = 1000
