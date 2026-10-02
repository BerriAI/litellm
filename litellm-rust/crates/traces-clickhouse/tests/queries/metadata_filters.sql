SELECT team_id AS team, api_key, request_id, spend,
       JSONExtractString(metadata, 'labels', 'priority') AS priority
FROM spend_logs FINAL
WHERE JSONExtractString(metadata, 'labels', 'priority') = 'high'
ORDER BY team, api_key, request_id
