SELECT
    request_id,
    JSONType(metadata, 'labels', 'priority') AS type,
    JSONExtractRaw(metadata, 'labels', 'priority') AS value
FROM calls
WHERE start_time >= now() - INTERVAL 1 DAY
    AND JSONHas(metadata, 'labels', 'priority')
LIMIT 100
