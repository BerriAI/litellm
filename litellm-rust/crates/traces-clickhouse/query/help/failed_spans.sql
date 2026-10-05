SELECT team_id AS team, api_key_hash AS api_key, trace_id,
       span_id, status_message AS message
FROM spans
WHERE start_time >= now() - INTERVAL 1 DAY AND status = 'error'
ORDER BY start_time DESC, team, api_key, trace_id, span_id
LIMIT 100
