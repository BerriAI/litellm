CREATE MATERIALIZED VIEW IF NOT EXISTS {database}.agent_traces_by_key_mv
TO {database}.agent_traces_by_key AS
SELECT
    TeamId, ApiKeyHash, TraceId,
    min(Timestamp)                                         AS StartTs,
    max(Timestamp + toIntervalNanosecond(Duration))        AS EndTs,
    any(ServiceName)                                       AS ServiceName,
    anyLastIf(toNullable(SpanName), ParentSpanId = '')     AS RootName,
    anyLastIf(toNullable(InputPreview), ParentSpanId = '') AS RootInput,
    anyLastIf(toNullable(StatusCode), ParentSpanId = '')   AS RootStatus,
    count()                                                AS SpanCount,
    countIf(ObservationType = 'agent')                     AS AgentCount,
    countIf(ObservationType = 'llm')                       AS LlmCount,
    countIf(ObservationType = 'tool')                      AS ToolCount,
    countIf(StatusCode = 'STATUS_CODE_ERROR')              AS ErrorCount,
    sum(InputTokens)                                       AS InputTokens,
    sum(OutputTokens)                                      AS OutputTokens,
    groupUniqArrayIf(toString(Model), Model != '')         AS Models,
    groupUniqArrayIf(SpanName, ObservationType = 'agent')  AS AgentNames,
    groupArrayIf(LiteLLMRequestId, LiteLLMRequestId != '') AS RequestIds
FROM {database}.otel_traces
GROUP BY TeamId, ApiKeyHash, TraceId
