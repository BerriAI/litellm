ALTER TABLE {database}.trace_rollup MODIFY TTL StartHour + INTERVAL {retention_days} DAY
