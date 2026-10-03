SELECT
    TraceId, SpanId, Model, InputTokens, OutputTokens,
    Duration / 1000000 AS duration_ms
FROM otel_traces
WHERE Timestamp >= now() - INTERVAL 1 DAY
    AND ObservationType = 'llm'
ORDER BY Timestamp DESC
LIMIT 100
