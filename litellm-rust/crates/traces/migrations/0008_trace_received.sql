ALTER TABLE {database}.otel_traces ADD COLUMN IF NOT EXISTS EngineReceivedMs UInt64 DEFAULT 0
