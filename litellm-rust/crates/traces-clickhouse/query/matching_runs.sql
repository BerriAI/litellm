run_keys AS (
    SELECT TeamId, ApiKeyHash, TraceId
    FROM owned_runs
    WHERE ReceivedMs <= {as_of_ms:UInt64}
      AND (({trace_id:String} != '' AND TraceId = {trace_id:String})
        OR ({trace_ref:String} != '' AND TraceRef = {trace_ref:String})
        OR ({trace_id:String} = '' AND {trace_ref:String} = ''
            AND StartHour >= toStartOfHour(fromUnixTimestamp64Milli({range_start_ms:Int64}, 'UTC'))
            AND StartHour < fromUnixTimestamp64Milli({range_end_ms:Int64}, 'UTC')
            AND (empty({trace_refs:Array(String)}) OR TraceRef IN {trace_refs:Array(String)})))
    GROUP BY TeamId, ApiKeyHash, TraceId
),
run_attributes AS (
    SELECT TeamId, ApiKeyHash, TraceId,
           groupUniqArrayArray(arrayFilter(i -> (mapContains(ResourceAttributes, {attribute_keys:Array(String)}[i]) AND ResourceAttributes[{attribute_keys:Array(String)}[i]] ILIKE {attribute_patterns:Array(String)}[i])
                                               OR (mapContains(SpanAttributes, {attribute_keys:Array(String)}[i]) AND SpanAttributes[{attribute_keys:Array(String)}[i]] ILIKE {attribute_patterns:Array(String)}[i]),
                                          arrayEnumerate({attribute_keys:Array(String)}))) AS matched_attributes
    FROM owned_spans
    WHERE notEmpty({attribute_keys:Array(String)})
      AND EngineReceivedMs <= {as_of_ms:UInt64}
      AND Timestamp >= fromUnixTimestamp64Milli({range_start_ms:Int64})
      AND (TeamId, ApiKeyHash, TraceId) IN run_keys
    GROUP BY TeamId, ApiKeyHash, TraceId
),
runs AS (
SELECT TraceId AS trace_id,
       any(TraceRef) AS trace_ref,
       if(uniqExact(UserId) = 1, any(UserId), '') AS user_id,
       TeamId AS team_id, ApiKeyHash AS api_key_hash,
       argMinMerge(Name) AS name,
       argMinMerge(Service) AS service,
       argMinMerge(RootInput) AS root_input,
       if(root_input != '', root_input, argMinMerge(AgentInput)) AS input_preview,
       argMinMerge(RootStatus) AS status,
       toUnixTimestamp64Milli(min(StartTs)) AS start_ms,
       toUInt64(greatest(toUnixTimestamp64Nano(max(EndTs)) - toUnixTimestamp64Nano(min(StartTs)), 0)) AS duration_ns,
       sum(SpanCount) AS span_count,
       sum(AgentSpans) AS agent_invocations,
       sum(LlmSpans) AS llm_calls,
       sum(ToolSpans) AS tool_calls,
       sum(InputTokens) AS input_tokens, sum(OutputTokens) AS output_tokens,
       groupUniqArrayArray(Models) AS models,
       sum(ErrorSpans) AS error_count,
       arraySort(groupUniqArrayArray(AgentLabels)) AS search_agents,
       multiIf(status = 'STATUS_CODE_OK', 'ok', status = 'STATUS_CODE_ERROR', 'error', 'unset') AS search_root_status,
       if(error_count > 0, 'true', 'false') AS search_has_error,
       length(groupUniqArrayArray(AgentIdentities)) AS agent_count,
       arraySort(groupUniqArrayArray(Frameworks)) AS frameworks,
       any(run_attributes.matched_attributes) AS matched_attributes
FROM trace_rollup
LEFT JOIN run_attributes USING (TeamId, ApiKeyHash, TraceId)
WHERE ReceivedMs <= {as_of_ms:UInt64}
  AND StartHour >= toStartOfHour(fromUnixTimestamp64Milli({rollups_from_ms:Int64}, 'UTC'))
  AND (TeamId, ApiKeyHash, TraceId) IN run_keys
GROUP BY TeamId, ApiKeyHash, TraceId
HAVING
