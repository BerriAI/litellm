ALTER TABLE {database}.agent_traces_by_key_mv MODIFY QUERY
SELECT
    TeamId, ApiKeyHash, TraceId, groupUniqArray(UserId) AS UserIds,
    min(Timestamp)                                         AS StartTs,
    max(Timestamp + toIntervalNanosecond(Duration))        AS EndTs,
    any(ServiceName)                                       AS ServiceName,
    anyLastIf(toNullable(SpanName), ParentSpanId = '')     AS RootName,
    anyLastIf(toNullable(InputPreview), ParentSpanId = '') AS RootInput,
    anyLastIf(toNullable(StatusCode), ParentSpanId = '')   AS RootStatus,
    count()                                                AS SpanCount,
    countIf(ObservationType = 'agent')                     AS AgentCount,
    countIf(ObservationType = 'llm')                       AS LlmCount,
    countIf(ObservationType = 'llm' AND LiteLLMRequestId != '') AS IdentifiedLlmCount,
    countIf(ObservationType = 'tool')                      AS ToolCount,
    countIf(StatusCode = 'STATUS_CODE_ERROR')              AS ErrorCount,
    sum(InputTokens)                                       AS InputTokens,
    sum(OutputTokens)                                      AS OutputTokens,
    groupUniqArrayIf(toString(Model), Model != '')         AS Models,
    groupUniqArrayIf(SpanName, ObservationType = 'agent')  AS AgentNames,
    groupArrayIf(LiteLLMRequestId, ObservationType = 'llm' OR LiteLLMRequestId != '') AS RequestIds
FROM {database}.otel_traces
GROUP BY TeamId, ApiKeyHash, TraceId
