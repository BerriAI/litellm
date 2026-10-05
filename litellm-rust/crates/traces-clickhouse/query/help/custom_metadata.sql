SELECT
    request_id, response_id, model, spend, JSONExtractString(metadata, 'project') AS project
FROM calls
WHERE start_time >= now() - INTERVAL 1 DAY
    AND JSONHas(metadata, 'project')
    AND JSONExtractString(metadata, 'project') = 'example'
ORDER BY start_time DESC
LIMIT 100
