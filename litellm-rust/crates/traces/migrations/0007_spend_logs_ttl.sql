ALTER TABLE {database}.spend_logs MODIFY TTL toDateTime(start_time) + INTERVAL {spend_log_retention_days} DAY
