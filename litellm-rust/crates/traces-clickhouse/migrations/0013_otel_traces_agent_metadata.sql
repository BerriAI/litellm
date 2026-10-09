ALTER TABLE {database}.otel_traces
    ADD COLUMN IF NOT EXISTS AgentMetadata String DEFAULT '{}' CODEC(ZSTD(3))
