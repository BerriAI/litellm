ALTER TABLE {database}.spend_logs ADD COLUMN IF NOT EXISTS EngineReceivedMs UInt64 DEFAULT 0
