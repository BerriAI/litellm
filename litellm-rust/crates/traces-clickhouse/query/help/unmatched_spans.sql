SELECT
    t.trace_id, t.span_id, t.model, t.request_id,
    t.input_tokens, t.output_tokens
FROM spans AS t
LEFT ANTI JOIN (
    SELECT *
    FROM calls
    WHERE start_time >= now() - INTERVAL 1 DAY
) AS s
    ON t.team_id = s.team_id
    AND ((t.user_id != '' AND t.user_id = s.user_id)
        OR (t.api_key_hash != '' AND t.api_key_hash = s.api_key_hash))
    AND t.request_id != ''
    AND (t.request_id = s.response_id OR t.request_id = s.request_id)
WHERE t.start_time >= now() - INTERVAL 1 DAY
    AND t.observation_type = 'llm'
ORDER BY t.start_time DESC, t.span_id
LIMIT 100
