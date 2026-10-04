SELECT TraceId AS trace_id,
       hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))) AS trace_ref,
       if(length(groupUniqArrayArray(UserIds)) = 1, arrayElement(groupUniqArrayArray(UserIds), 1), '') AS user_id, TeamId AS team_id, ApiKeyHash AS api_key_hash,
       ifNull(any(RootName), '') AS name, any(ServiceName) AS service,
       ifNull(any(RootInput), '') AS input_preview, ifNull(any(RootStatus), '') AS status,
       toUnixTimestamp64Milli(min(StartTs)) AS start_ms,
       min(StartTs) AS trace_start, max(EndTs) AS trace_end,
       dateDiff('millisecond', min(StartTs), max(EndTs)) AS duration_ms,
       sum(SpanCount) AS span_count,
       sum(AgentCount) AS agent_invocations,
       sum(LlmCount) AS llm_calls, sum(ToolCount) AS tool_calls,
       sum(InputTokens) AS input_tokens, sum(OutputTokens) AS output_tokens,
       groupUniqArrayArray(Models) AS models, sum(ErrorCount) AS error_count,
       arraySort(if(empty(groupUniqArrayArray(AgentLabels)),
                    groupUniqArrayArray(AgentNames),
                    groupUniqArrayArray(AgentLabels))) AS search_agents,
       if(error_count > 0, 'error', 'ok') AS search_status
FROM owned_runs
WHERE {trace_id:String} = '' OR TraceId = {trace_id:String}
GROUP BY TeamId, ApiKeyHash, TraceId
HAVING {trace_id:String} != ''
    OR (min(StartTs) >= fromUnixTimestamp64Milli({start_ms:Int64})
        AND min(StartTs) < fromUnixTimestamp64Milli({end_ms:Int64})
        AND arrayAll(t -> trace_id ILIKE t OR input_preview ILIKE t OR name ILIKE t, {text:Array(String)})
        AND arrayAll((f, p, m) -> (m = 'exclude') != multiIf(
                f = 'name', name ILIKE p,
                f = 'agent', arrayExists(a -> a ILIKE p, search_agents),
                f = 'status', search_status ILIKE p,
                f = 'model', arrayExists(x -> x ILIKE p, models),
                f = 'input', input_preview ILIKE p,
                f = 'trace_id', trace_id ILIKE p,
                false),
              {filter_fields:Array(String)}, {filter_patterns:Array(String)}, {filter_modes:Array(String)}))
