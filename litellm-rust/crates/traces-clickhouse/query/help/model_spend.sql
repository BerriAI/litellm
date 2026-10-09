SELECT
    team_id, model, requests, unknown_cost_requests,
    if(unknown_cost_requests = 0, recorded_spend, NULL) AS spend,
    input_tokens, output_tokens
FROM (
    SELECT
        team_id, model, count() AS requests,
        countIf(isNull(spend) OR NOT isFinite(spend)) AS unknown_cost_requests,
        sum(spend) AS recorded_spend,
        sum(prompt_tokens) AS input_tokens,
        sum(completion_tokens) AS output_tokens
    FROM spend_logs FINAL
    WHERE start_time >= now() - INTERVAL 1 DAY
    GROUP BY team_id, model
)
ORDER BY team_id, model
LIMIT 100
