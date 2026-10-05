SELECT team_id AS team, api_key_hash AS api_key, request_id, spend,
       JSONExtractString(metadata, 'labels', 'priority') AS priority
FROM calls
WHERE start_time >= now() - INTERVAL 1 DAY
    AND JSONHas(metadata, 'labels', 'priority')
    AND JSONExtractString(metadata, 'labels', 'priority') = 'high'
ORDER BY team, api_key, request_id
LIMIT 100
