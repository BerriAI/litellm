ALTER TABLE {database}.agent_traces_by_key MODIFY TTL toDateTime(StartTs) + INTERVAL {retention_days} DAY
