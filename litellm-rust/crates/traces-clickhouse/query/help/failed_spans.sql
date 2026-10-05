SELECT TeamId AS team, ApiKeyHash AS api_key, TraceId AS trace_id,
       SpanId AS span_id, StatusMessage AS message
FROM (
    SELECT *
    FROM otel_traces
    WHERE Timestamp >= now() - INTERVAL 1 DAY
    ORDER BY Timestamp, EngineReceivedMs, StatusMessage, Duration, StatusCode
    LIMIT 1 BY TeamId, ApiKeyHash, TraceId, SpanId
)
WHERE StatusCode = 'STATUS_CODE_ERROR'
ORDER BY Timestamp DESC, team, api_key, trace_id, span_id
LIMIT 100
