ALTER TABLE {database}.otel_traces MODIFY TTL toDateTime(Timestamp) + INTERVAL {trace_retention_days} DAY
