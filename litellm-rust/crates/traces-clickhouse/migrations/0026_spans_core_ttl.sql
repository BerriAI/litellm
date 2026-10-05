ALTER TABLE {database}.spans_core MODIFY TTL toDateTime(Timestamp) + INTERVAL {retention_days} DAY
