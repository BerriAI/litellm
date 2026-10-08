ALTER TABLE {database}.otel_traces
    ADD COLUMN IF NOT EXISTS UserId String DEFAULT ''
