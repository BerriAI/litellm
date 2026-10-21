SELECT
    team_id, api_key, trace_id, count() AS requests,
    countIf(isNull(spend) OR NOT isFinite(spend)) AS unknown_cost_requests,
    if(unknown_cost_requests = 0, sum(spend), NULL) AS recorded_spend
FROM spend_logs FINAL
WHERE start_time >= now() - INTERVAL 1 DAY
    AND trace_id != ''
GROUP BY team_id, api_key, trace_id
ORDER BY team_id, api_key, trace_id
LIMIT 100
