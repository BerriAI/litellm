ALTER TABLE {database}.spend_logs
    ADD COLUMN IF NOT EXISTS provider_request_id String DEFAULT '' AFTER response_id,
    ADD INDEX IF NOT EXISTS idx_provider_request_id provider_request_id
        TYPE bloom_filter(0.001) GRANULARITY 1
