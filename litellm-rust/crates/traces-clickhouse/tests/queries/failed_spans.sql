SELECT TeamId AS team, ApiKeyHash AS api_key, TraceId AS trace_id,
       SpanId AS span_id, StatusMessage AS message
FROM otel_traces
WHERE StatusCode = 'STATUS_CODE_ERROR'
ORDER BY team, api_key, trace_id, span_id
