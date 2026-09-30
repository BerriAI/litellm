ALTER TABLE {database}.agent_traces MODIFY TTL toDateTime(StartTs) + INTERVAL {trace_retention_days} DAY
