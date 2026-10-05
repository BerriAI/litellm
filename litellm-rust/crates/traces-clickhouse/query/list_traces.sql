WITH page AS (
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
       arrayDistinct(if(sum(IdentifiedLlmCount) != sum(LlmCount),
                        arrayConcat(groupArrayArray(RequestIds), ['']),
                        groupArrayArray(RequestIds))) AS request_ids
FROM agent_traces_by_key
WHERE ({all_teams:UInt8} = 1
       OR ({user_id:String} != '' AND UserIds = [{user_id:String}])
       OR has({team_ids:Array(String)}, TeamId))
GROUP BY TeamId, ApiKeyHash, TraceId
HAVING min(StartTs) >= fromUnixTimestamp64Milli({start_ms:Int64})
   AND min(StartTs) < fromUnixTimestamp64Milli({end_ms:Int64})
   AND ({cursor_ms:Int64} = 0 OR (toUnixTimestamp64Milli(min(StartTs)), trace_ref)
        < ({cursor_ms:Int64}, {cursor_trace_id:String}))
ORDER BY start_ms DESC, trace_ref DESC
LIMIT {limit:UInt32}
)
SELECT page.* EXCEPT (trace_start, trace_end),
       identities.agent_names AS agent_names, identities.agent_count AS agent_count,
       identities.frameworks AS frameworks
FROM page
LEFT JOIN (
    SELECT TeamId, ApiKeyHash, TraceId,
           arraySort(groupUniqArrayIf(AgentName, AgentName != '')) AS agent_names,
           arraySort(groupUniqArrayIf(toString(Framework), Framework != '')) AS frameworks,
           uniqExactIf(if(AgentName = '', SpanName, AgentName), ObservationType = 'agent') AS agent_count
    FROM otel_traces
    WHERE Timestamp >= (SELECT min(trace_start) FROM page)
      AND Timestamp <= (SELECT max(trace_end) FROM page)
      AND TraceId IN (SELECT trace_id FROM page)
      AND (TeamId, ApiKeyHash, TraceId) IN (SELECT team_id, api_key_hash, trace_id FROM page)
    GROUP BY TeamId, ApiKeyHash, TraceId
) AS identities
ON page.team_id = identities.TeamId AND page.api_key_hash = identities.ApiKeyHash
   AND page.trace_id = identities.TraceId
ORDER BY page.start_ms DESC, page.trace_ref DESC
