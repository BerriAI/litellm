SELECT team_id AS team, api_key, request_id, spend,
       JSONExtractString(metadata, 'labels', 'priority') AS priority
FROM spend_logs FINAL
WHERE start_time >= now() - INTERVAL 1 DAY
    AND JSONHas(metadata, 'labels', 'priority')
    AND JSONExtractString(metadata, 'labels', 'priority') = 'high'
ORDER BY team, api_key, request_id
LIMIT 100
