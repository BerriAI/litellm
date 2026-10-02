ALTER TABLE {database}.otel_traces MODIFY TTL toDateTime(Timestamp) + INTERVAL {retention_days} DAY
