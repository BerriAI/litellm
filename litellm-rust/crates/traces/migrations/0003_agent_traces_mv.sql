CREATE MATERIALIZED VIEW IF NOT EXISTS {database}.agent_traces_mv TO {database}.agent_traces AS
SELECT
    TeamId, TraceId,
    min(Timestamp)                                         AS StartTs,
    max(Timestamp + toIntervalNanosecond(Duration))        AS EndTs,
    any(ServiceName)                                       AS ServiceName,
    anyLastIf(SpanName, ParentSpanId = '')                 AS RootName,
    anyLastIf(InputPreview, ParentSpanId = '')             AS RootInput,
    anyLastIf(StatusCode, ParentSpanId = '')               AS RootStatus,
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
GROUP BY TeamId, TraceId
