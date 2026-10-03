SELECT
    t.TraceId, t.SpanId, t.Model, t.LiteLLMRequestId,
    t.InputTokens, t.OutputTokens
FROM otel_traces AS t
LEFT ANTI JOIN (
    SELECT *
    FROM spend_logs FINAL
    WHERE start_time >= now() - INTERVAL 1 DAY
) AS s
    ON t.TeamId = s.team_id
    AND ((t.UserId != '' AND t.UserId = s.user)
        OR (t.ApiKeyHash != '' AND t.ApiKeyHash = s.api_key))
    AND t.LiteLLMRequestId != ''
    AND (t.LiteLLMRequestId = s.response_id OR t.LiteLLMRequestId = s.request_id)
WHERE t.Timestamp >= now() - INTERVAL 1 DAY
    AND t.ObservationType = 'llm'
ORDER BY t.Timestamp DESC, t.SpanId
LIMIT 100
