SELECT t.TraceId, t.SpanId, s.request_id, s.spend, s.metadata
FROM otel_traces AS t
INNER JOIN (
    SELECT *
    FROM spend_logs FINAL
    WHERE start_time >= now() - INTERVAL 1 DAY
) AS s
    ON t.LiteLLMRequestId = s.response_id
    AND t.TeamId = s.team_id
    AND ((t.UserId != '' AND t.UserId = s.user)
        OR (t.ApiKeyHash != '' AND t.ApiKeyHash = s.api_key))
WHERE t.Timestamp >= now() - INTERVAL 1 DAY
    AND t.LiteLLMRequestId != ''
LIMIT 100
