ALTER TABLE {database}.otel_traces ADD COLUMN IF NOT EXISTS Framework LowCardinality(String) AFTER AgentName
