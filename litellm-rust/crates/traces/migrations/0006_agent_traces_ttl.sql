ALTER TABLE {database}.agent_traces_by_key MODIFY TTL toDateTime(StartTs) + INTERVAL {trace_retention_days} DAY
