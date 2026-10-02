WITH page AS (
SELECT TraceId AS trace_id,
       hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))) AS trace_ref,
       TeamId AS team_id, ApiKeyHash AS api_key_hash,
       ifNull(any(RootName), '') AS name, any(ServiceName) AS service,
       ifNull(any(RootInput), '') AS input_preview, ifNull(any(RootStatus), '') AS status,
       toUnixTimestamp64Milli(min(StartTs)) AS start_ms,
       dateDiff('millisecond', min(StartTs), max(EndTs)) AS duration_ms,
       sum(SpanCount) AS span_count, length(groupUniqArrayArray(AgentNames)) AS agent_count,
       sum(AgentCount) AS agent_invocations,
       sum(LlmCount) AS llm_calls, sum(ToolCount) AS tool_calls,
       sum(InputTokens) AS input_tokens, sum(OutputTokens) AS output_tokens,
       groupUniqArrayArray(Models) AS models, sum(ErrorCount) AS error_count,
       arrayDistinct(groupArrayArray(RequestIds)) AS request_ids
FROM agent_traces_by_key
WHERE (empty({team_ids:Array(String)}) OR TeamId IN {team_ids:Array(String)})
  AND ({api_key_hash:String} = '' OR ApiKeyHash = {api_key_hash:String})
GROUP BY TeamId, ApiKeyHash, TraceId
HAVING min(StartTs) >= fromUnixTimestamp64Milli({start_ms:Int64})
   AND min(StartTs) < fromUnixTimestamp64Milli({end_ms:Int64})
   AND ({cursor_ms:Int64} = 0 OR (toUnixTimestamp64Milli(min(StartTs)), trace_ref)
        < ({cursor_ms:Int64}, {cursor_trace_id:String}))
ORDER BY start_ms DESC, trace_ref DESC
LIMIT {limit:UInt32}
)
SELECT page.*, identities.agent_names AS agent_names
FROM page
LEFT JOIN (
    SELECT TeamId, ApiKeyHash, TraceId, arraySort(groupUniqArray(AgentName)) AS agent_names
    FROM otel_traces
    WHERE AgentName != ''
      AND TraceId IN (SELECT trace_id FROM page)
      AND (TeamId, ApiKeyHash, TraceId) IN (SELECT team_id, api_key_hash, trace_id FROM page)
    GROUP BY TeamId, ApiKeyHash, TraceId
) AS identities
ON page.team_id = identities.TeamId AND page.api_key_hash = identities.ApiKeyHash
   AND page.trace_id = identities.TraceId
ORDER BY page.start_ms DESC, page.trace_ref DESC
