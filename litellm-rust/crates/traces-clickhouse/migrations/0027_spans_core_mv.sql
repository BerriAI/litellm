CREATE MATERIALIZED VIEW IF NOT EXISTS {database}.spans_core_mv
TO {database}.spans_core AS
SELECT TeamId, ApiKeyHash, UserId, TraceId, SpanId, ParentSpanId,
       Timestamp, Duration, EngineReceivedMs,
       SpanName, ServiceName, StatusCode, StatusMessage,
       ObservationType, AgentName, Framework, Model, InputTokens, OutputTokens, InputPreview
FROM {database}.otel_traces
