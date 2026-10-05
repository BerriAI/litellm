run_keys AS (
    SELECT TeamId, ApiKeyHash, TraceId
    FROM owned_core
    WHERE EngineReceivedMs <= {as_of_ms:UInt64}
      AND (({trace_id:String} != '' AND TraceId = {trace_id:String})
        OR ({trace_ref:String} != '' AND TraceRef = {trace_ref:String})
        OR ({trace_id:String} = '' AND {trace_ref:String} = ''
            AND Timestamp >= fromUnixTimestamp64Milli({range_start_ms:Int64})
            AND Timestamp < fromUnixTimestamp64Milli({range_end_ms:Int64})
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
run_spans AS (
    SELECT * FROM spans_core
    WHERE EngineReceivedMs <= {as_of_ms:UInt64}
      AND TraceId IN (SELECT TraceId FROM run_keys)
      AND (TeamId, ApiKeyHash, TraceId) IN run_keys
    ORDER BY Timestamp, EngineReceivedMs, StatusMessage, Duration, StatusCode
    LIMIT 1 BY TeamId, ApiKeyHash, TraceId, SpanId
),
runs AS (
SELECT TraceId AS trace_id,
       any(TraceRef) AS trace_ref,
       if(uniqExact(UserId) = 1, any(UserId), '') AS user_id,
       TeamId AS team_id, ApiKeyHash AS api_key_hash,
       argMin(CAST(SpanName, 'String'), (toUInt8(ParentSpanId != ''), Timestamp, SpanId)) AS name,
       argMin(CAST(ServiceName, 'String'), (Timestamp, SpanId)) AS service,
       argMin(InputPreview, (toUInt8(ParentSpanId != ''), Timestamp, SpanId)) AS root_input,
       if(root_input != '', root_input,
          argMinIf(InputPreview, (Timestamp, SpanId), InputPreview != '' AND ObservationType IN ('agent', 'llm'))) AS input_preview,
       argMin(CAST(StatusCode, 'String'), (toUInt8(ParentSpanId != ''), Timestamp, SpanId)) AS status,
       toUnixTimestamp64Milli(min(Timestamp)) AS start_ms,
       toUInt64(greatest(toUnixTimestamp64Nano(max(Timestamp + toIntervalNanosecond(Duration))) - toUnixTimestamp64Nano(min(Timestamp)), 0)) AS duration_ns,
       count() AS span_count,
       countIf(ObservationType = 'agent') AS agent_invocations,
       countIf(ObservationType = 'llm') AS llm_calls,
       countIf(ObservationType = 'tool') AS tool_calls,
       sum(InputTokens) AS input_tokens, sum(OutputTokens) AS output_tokens,
       groupUniqArrayIf(CAST(Model, 'String'), Model != '') AS models,
       countIf(StatusCode = 'STATUS_CODE_ERROR') AS error_count,
       arraySort(groupUniqArrayIf(if(AgentName = '', CAST(SpanName, 'String'), CAST(AgentName, 'String')),
                                  AgentName != '' OR ObservationType = 'agent')) AS search_agents,
       multiIf(status = 'STATUS_CODE_OK', 'ok', status = 'STATUS_CODE_ERROR', 'error', 'unset') AS search_root_status,
       if(error_count > 0, 'true', 'false') AS search_has_error,
       uniqExactIf(if(AgentName = '', CAST(SpanName, 'String'), CAST(AgentName, 'String')), ObservationType = 'agent') AS agent_count,
       arraySort(groupUniqArrayIf(CAST(Framework, 'String'), Framework != '')) AS frameworks,
       any(run_attributes.matched_attributes) AS matched_attributes
FROM run_spans
LEFT JOIN run_attributes USING (TeamId, ApiKeyHash, TraceId)
GROUP BY TeamId, ApiKeyHash, TraceId
HAVING
