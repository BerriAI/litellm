INSERT INTO {database}.spans_core
SELECT TeamId, ApiKeyHash, UserId, TraceId, SpanId, ParentSpanId,
       hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))) AS TraceRef,
       Timestamp, Duration, EngineReceivedMs,
       SpanName, ServiceName, StatusCode, StatusMessage,
       ObservationType, AgentName, Framework, Model, InputTokens, OutputTokens, InputPreview
FROM {database}.otel_traces
