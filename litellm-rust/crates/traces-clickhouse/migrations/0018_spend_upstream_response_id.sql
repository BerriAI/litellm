ALTER TABLE {database}.spend_logs
    ADD COLUMN IF NOT EXISTS upstream_response_id String MATERIALIZED
        if(startsWith(response_id, 'resp_'),
           extract(tryBase64Decode(substring(response_id, 6)), 'response_id:([^;]+)'),
           '') AFTER response_id,
    ADD INDEX IF NOT EXISTS idx_upstream_response_id upstream_response_id
        TYPE bloom_filter(0.001) GRANULARITY 1
