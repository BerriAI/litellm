CREATE TABLE IF NOT EXISTS {database}.lens_feedback
(
    id String,
    trace_id String,
    trace_ref String,
    span_id String,
    author_id String,
    value UInt8,
    payload String CODEC(ZSTD(3)),
    version UInt128,
    deleted UInt8 DEFAULT 0,
    CONSTRAINT usefulness_range CHECK value <= 10
)
ENGINE = ReplacingMergeTree(version)
ORDER BY (id)
