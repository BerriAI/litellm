WITH candidates AS (
SELECT TeamId, ApiKeyHash, TraceId
FROM agent_traces_by_key
WHERE StartTs >= fromUnixTimestamp64Milli({start_ms:Int64})
  AND StartTs < fromUnixTimestamp64Milli({end_ms:Int64})
  AND ({cursor_ms:Int64} = 0 OR toUnixTimestamp64Milli(StartTs) <= {cursor_ms:Int64})
  AND ({all_teams:UInt8} = 1
       OR ({user_id:String} != '' AND UserIds = [{user_id:String}])
       OR has({team_ids:Array(String)}, TeamId))
),
keys AS (
SELECT TeamId, ApiKeyHash, TraceId, min(StartTs) AS trace_start,
       hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))) AS trace_ref
FROM agent_traces_by_key
WHERE (TeamId, ApiKeyHash, TraceId) IN candidates
  AND ({all_teams:UInt8} = 1
       OR ({user_id:String} != '' AND UserIds = [{user_id:String}])
       OR has({team_ids:Array(String)}, TeamId))
GROUP BY TeamId, ApiKeyHash, TraceId
HAVING trace_start >= fromUnixTimestamp64Milli({start_ms:Int64})
   AND ({cursor_ms:Int64} = 0 OR (toUnixTimestamp64Milli(trace_start), trace_ref)
        < ({cursor_ms:Int64}, {cursor_trace_id:String}))
ORDER BY trace_start DESC, trace_ref DESC
LIMIT {limit:UInt32}
),
page AS (
SELECT TraceId AS trace_id,
       hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))) AS trace_ref,
       if(length(groupUniqArrayArray(UserIds)) = 1, arrayElement(groupUniqArrayArray(UserIds), 1), '') AS user_id, TeamId AS team_id, ApiKeyHash AS api_key_hash,
       ifNull(any(RootName), '') AS name, any(ServiceName) AS service,
       ifNull(any(RootInput), '') AS input_preview, ifNull(any(RootStatus), '') AS status,
       toUnixTimestamp64Milli(min(StartTs)) AS start_ms,
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
WHERE (TeamId, ApiKeyHash, TraceId) IN (SELECT TeamId, ApiKeyHash, TraceId FROM keys)
  AND ({all_teams:UInt8} = 1
       OR ({user_id:String} != '' AND UserIds = [{user_id:String}])
       OR has({team_ids:Array(String)}, TeamId))
GROUP BY TeamId, ApiKeyHash, TraceId
)
SELECT page.*,
       identities.agent_names AS agent_names, identities.agent_count AS agent_count,
       identities.frameworks AS frameworks
FROM page
LEFT JOIN (
    SELECT TeamId, ApiKeyHash, TraceId,
           arraySort(groupUniqArrayIf(AgentName, AgentName != '')) AS agent_names,
           arraySort(groupUniqArrayIf(toString(Framework), Framework != '')) AS frameworks,
           uniqExactIf(if(AgentName = '', SpanName, AgentName), ObservationType = 'agent') AS agent_count
    FROM otel_traces
    WHERE Timestamp >= fromUnixTimestamp64Milli({start_ms:Int64})
      AND TraceId IN (SELECT trace_id FROM page)
    GROUP BY TeamId, ApiKeyHash, TraceId
) AS identities
ON page.team_id = identities.TeamId AND page.api_key_hash = identities.ApiKeyHash
   AND page.trace_id = identities.TraceId
ORDER BY page.start_ms DESC, page.trace_ref DESC
