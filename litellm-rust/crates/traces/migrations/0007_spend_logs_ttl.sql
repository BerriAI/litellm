ALTER TABLE {database}.spend_logs MODIFY TTL toDateTime(start_time) + INTERVAL {retention_days} DAY
