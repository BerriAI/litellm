CREATE OR REPLACE VIEW {database}.traces
SQL SECURITY INVOKER
AS SELECT
    TeamId AS team_id,
    ApiKeyHash AS api_key_hash,
    TraceId AS trace_id,
    any(TraceRef) AS id,
    if(uniqExact(UserId) = 1, any(UserId), '') AS user_id,
    argMin(CAST(SpanName, 'String'), (toUInt8(ParentSpanId != ''), Timestamp, SpanId)) AS name,
    argMin(CAST(ServiceName, 'String'), (Timestamp, SpanId)) AS service,
    argMin(InputPreview, (toUInt8(ParentSpanId != ''), Timestamp, SpanId)) AS root_input,
    argMinIf(InputPreview, (Timestamp, SpanId), InputPreview != '' AND ObservationType IN ('agent', 'llm')) AS agent_input,
    if(root_input != '', root_input, agent_input) AS input_preview,
    argMin(CAST(StatusCode, 'String'), (toUInt8(ParentSpanId != ''), Timestamp, SpanId)) AS root_status_code,
    multiIf(root_status_code = 'STATUS_CODE_ERROR', 'error',
            root_status_code = 'STATUS_CODE_OK', 'ok', 'unset') AS root_status,
    min(Timestamp) AS start_time,
    max(Timestamp + toIntervalNanosecond(Duration)) AS end_time,
    toUInt64(greatest(toUnixTimestamp64Nano(end_time) - toUnixTimestamp64Nano(start_time), 0)) AS duration_ns,
    duration_ns / 1000000.0 AS duration_ms,
    count() AS span_count,
    countIf(StatusCode = 'STATUS_CODE_ERROR') AS error_count,
    CAST(error_count > 0, 'Bool') AS has_error,
    countIf(ObservationType = 'agent') AS agent_span_count,
    arraySort(groupUniqArrayIf(if(AgentName = '', CAST(SpanName, 'String'), CAST(AgentName, 'String')),
                               AgentName != '' OR ObservationType = 'agent')) AS agent_names,
    length(agent_names) AS agent_label_count,
    arraySort(groupUniqArrayIf(CAST(Framework, 'String'), Framework != '')) AS frameworks,
    countIf(ObservationType = 'llm') AS llm_span_count,
    countIf(ObservationType = 'tool') AS tool_span_count,
    sum(InputTokens) AS span_input_tokens,
    sum(OutputTokens) AS span_output_tokens,
    arraySort(groupUniqArrayIf(CAST(Model, 'String'), Model != '')) AS models
FROM (
    SELECT * FROM {database}.spans_core
    ORDER BY Timestamp, EngineReceivedMs, StatusMessage, Duration, StatusCode
    LIMIT 1 BY TeamId, ApiKeyHash, TraceId, SpanId
)
GROUP BY TeamId, ApiKeyHash, TraceId
