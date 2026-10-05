SELECT TraceId AS trace_id,
       hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))) AS trace_ref,
       if(length(groupUniqArray(UserId)) = 1, arrayElement(groupUniqArray(UserId), 1), '') AS user_id,
       TeamId AS team_id, ApiKeyHash AS api_key_hash,
       if(countIf(ParentSpanId = '') > 0, argMinIf(SpanName, tuple(Timestamp, SpanId), ParentSpanId = ''), argMin(SpanName, tuple(Timestamp, SpanId))) AS name,
       argMin(ServiceName, tuple(Timestamp, SpanId)) AS service,
       if(countIf(ParentSpanId = '') > 0, argMinIf(InputPreview, tuple(Timestamp, SpanId), ParentSpanId = ''), argMin(InputPreview, tuple(Timestamp, SpanId))) AS root_input,
       if(root_input != '', root_input, argMinIf(InputPreview, tuple(Timestamp, SpanId), InputPreview != '' AND ObservationType IN ('agent', 'llm'))) AS input_preview,
       if(countIf(ParentSpanId = '') > 0, argMinIf(StatusCode, tuple(Timestamp, SpanId), ParentSpanId = ''), argMin(StatusCode, tuple(Timestamp, SpanId))) AS status,
       toUnixTimestamp64Milli(min(Timestamp)) AS start_ms,
       toUInt64(greatest(toUnixTimestamp64Nano(max(Timestamp + toIntervalNanosecond(Duration))) - toUnixTimestamp64Nano(min(Timestamp)), 0)) AS duration_ns,
       count() AS span_count,
       countIf(ObservationType = 'agent') AS agent_invocations,
       countIf(ObservationType = 'llm') AS llm_calls,
       countIf(ObservationType = 'tool') AS tool_calls,
       sum(InputTokens) AS input_tokens, sum(OutputTokens) AS output_tokens,
       groupUniqArrayIf(toString(Model), Model != '') AS models,
       countIf(StatusCode = 'STATUS_CODE_ERROR') AS error_count,
       arraySort(groupUniqArrayIf(if(AgentName = '', SpanName, AgentName), AgentName != '' OR ObservationType = 'agent')) AS search_agents,
       multiIf(status = 'STATUS_CODE_OK', 'ok', status = 'STATUS_CODE_ERROR', 'error', 'unset') AS search_root_status,
       if(error_count > 0, 'true', 'false') AS search_has_error,
       length(groupUniqArrayIf(if(AgentName = '', SpanName, AgentName), ObservationType = 'agent')) AS agent_count,
       arraySort(groupUniqArrayIf(toString(Framework), Framework != '')) AS frameworks,
       groupUniqArrayArray(arrayFilter(i -> (mapContains(ResourceAttributes, {attribute_keys:Array(String)}[i]) AND ResourceAttributes[{attribute_keys:Array(String)}[i]] ILIKE {attribute_patterns:Array(String)}[i])
                                           OR (mapContains(SpanAttributes, {attribute_keys:Array(String)}[i]) AND SpanAttributes[{attribute_keys:Array(String)}[i]] ILIKE {attribute_patterns:Array(String)}[i]),
                                      arrayEnumerate({attribute_keys:Array(String)}))) AS matched_attributes
FROM canonical_spans
GROUP BY TeamId, ApiKeyHash, TraceId
HAVING ({trace_id:String} != '' AND TraceId = {trace_id:String})
    OR ({trace_ref:String} != '' AND trace_ref = {trace_ref:String})
    OR ({trace_id:String} = '' AND {trace_ref:String} = ''
        AND min(Timestamp) >= fromUnixTimestamp64Milli({start_ms:Int64})
        AND min(Timestamp) < fromUnixTimestamp64Milli({end_ms:Int64})
        AND arrayAll(t -> trace_id ILIKE t OR input_preview ILIKE t OR name ILIKE t, {text:Array(String)})
        AND arrayAll((f, p, m) -> (m = 'exclude') != multiIf(
                f = 'name', name ILIKE p,
                f = 'agent', arrayExists(a -> a ILIKE p, search_agents),
                f = 'root_status', search_root_status ILIKE p,
                f = 'has_error', search_has_error ILIKE p,
                f = 'model', arrayExists(x -> x ILIKE p, models),
                f = 'input', input_preview ILIKE p,
                f = 'trace_id', trace_id ILIKE p,
                f = 'service', service ILIKE p,
                f = 'team', team_id ILIKE p,
                false),
              {filter_fields:Array(String)}, {filter_patterns:Array(String)}, {filter_modes:Array(String)})
        AND arrayAll((i, m) -> (m = 'exclude') != has(matched_attributes, i),
              arrayEnumerate({attribute_keys:Array(String)}), {attribute_modes:Array(String)})
        AND (empty({trace_refs:Array(String)}) OR trace_ref IN {trace_refs:Array(String)}))
