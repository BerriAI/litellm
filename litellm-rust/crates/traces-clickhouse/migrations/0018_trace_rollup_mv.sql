CREATE MATERIALIZED VIEW IF NOT EXISTS {database}.trace_rollup_mv
TO {database}.trace_rollup AS
SELECT
    TeamId, toStartOfHour(toDateTime(min(Timestamp), 'UTC')) AS StartHour, ApiKeyHash, TraceId, UserId,
    hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))) AS TraceRef,
    min(EngineReceivedMs) AS ReceivedMs,
    min(Timestamp) AS StartTs,
    max(Timestamp + toIntervalNanosecond(Duration)) AS EndTs,
    argMinState(toString(SpanName), (toUInt8(ParentSpanId != ''), Timestamp, SpanId)) AS Name,
    argMinState(toString(ServiceName), (Timestamp, SpanId)) AS Service,
    argMinState(InputPreview, (toUInt8(ParentSpanId != ''), Timestamp, SpanId)) AS RootInput,
    argMinState(if(InputPreview != '' AND ObservationType IN ('agent', 'llm'), InputPreview, ''),
                (toUInt8(InputPreview = '' OR ObservationType NOT IN ('agent', 'llm')), Timestamp, SpanId)) AS AgentInput,
    argMinState(toString(StatusCode), (toUInt8(ParentSpanId != ''), Timestamp, SpanId)) AS RootStatus,
    count() AS SpanCount,
    countIf(ObservationType = 'agent') AS AgentSpans,
    countIf(ObservationType = 'llm') AS LlmSpans,
    countIf(ObservationType = 'tool') AS ToolSpans,
    countIf(StatusCode = 'STATUS_CODE_ERROR') AS ErrorSpans,
    sum(InputTokens) AS InputTokens,
    sum(OutputTokens) AS OutputTokens,
    groupUniqArrayIf(toString(Model), Model != '') AS Models,
    groupUniqArrayIf(if(AgentName = '', toString(SpanName), toString(AgentName)), AgentName != '' OR ObservationType = 'agent') AS AgentLabels,
    groupUniqArrayIf(if(AgentName = '', toString(SpanName), toString(AgentName)), ObservationType = 'agent') AS AgentIdentities,
    groupUniqArrayIf(toString(Framework), Framework != '') AS Frameworks
FROM (
    SELECT * FROM {database}.otel_traces
    ORDER BY Timestamp, EngineReceivedMs, StatusMessage, Duration, StatusCode
    LIMIT 1 BY TeamId, ApiKeyHash, TraceId, SpanId
)
GROUP BY TeamId, ApiKeyHash, TraceId, UserId
