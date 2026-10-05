CREATE OR REPLACE VIEW {database}.traces
SQL SECURITY INVOKER
AS SELECT
    TeamId AS team_id,
    ApiKeyHash AS api_key_hash,
    TraceId AS trace_id,
    any(TraceRef) AS id,
    if(uniqExact(UserId) = 1, any(UserId), '') AS user_id,
    argMinMerge(Name) AS name,
    argMinMerge(Service) AS service,
    if(argMinMerge(RootInput) != '', argMinMerge(RootInput), argMinMerge(AgentInput)) AS input_preview,
    multiIf(argMinMerge(RootStatus) = 'STATUS_CODE_ERROR', 'error',
            argMinMerge(RootStatus) = 'STATUS_CODE_OK', 'ok', 'unset') AS root_status,
    min(StartTs) AS start_time,
    max(EndTs) AS end_time,
    toUInt64(greatest(toUnixTimestamp64Nano(end_time) - toUnixTimestamp64Nano(start_time), 0)) AS duration_ns,
    duration_ns / 1000000.0 AS duration_ms,
    sum(SpanCount) AS span_count,
    sum(ErrorSpans) AS error_count,
    CAST(error_count > 0, 'Bool') AS has_error,
    sum(AgentSpans) AS agent_span_count,
    arraySort(groupUniqArrayArray(AgentLabels)) AS agent_names,
    length(agent_names) AS agent_label_count,
    arraySort(groupUniqArrayArray(Frameworks)) AS frameworks,
    sum(LlmSpans) AS llm_span_count,
    sum(ToolSpans) AS tool_span_count,
    sum(InputTokens) AS span_input_tokens,
    sum(OutputTokens) AS span_output_tokens,
    arraySort(groupUniqArrayArray(Models)) AS models
FROM {database}.trace_rollup
GROUP BY TeamId, ApiKeyHash, TraceId
