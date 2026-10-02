SELECT o.TeamId AS team, o.ApiKeyHash AS api_key, o.TraceId AS trace_id,
       sum(s.spend) AS spend
FROM otel_traces AS o
INNER JOIN (SELECT * FROM spend_logs FINAL) AS s
    ON o.TeamId = s.team_id
   AND o.ApiKeyHash = s.api_key
   AND o.LiteLLMRequestId = s.response_id
GROUP BY o.TeamId, o.ApiKeyHash, o.TraceId
ORDER BY team, api_key, trace_id
