SELECT
    DISTINCT arrayJoin(JSONExtractKeys(metadata)) AS key
FROM calls
WHERE start_time >= now() - INTERVAL 30 DAY
ORDER BY key
LIMIT 200
