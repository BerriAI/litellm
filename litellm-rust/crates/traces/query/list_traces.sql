SELECT t.TraceId AS trace_id, any(t.RootName) AS name, any(t.ServiceName) AS service,
       any(t.RootInput) AS input_preview, any(t.RootStatus) AS status,
       toUnixTimestamp64Milli(any(t.StartTs)) AS start_ms,
       dateDiff('millisecond', any(t.StartTs), any(t.EndTs)) AS duration_ms,
       any(t.SpanCount) AS span_count, length(any(t.AgentNames)) AS agent_count,
       any(t.AgentCount) AS agent_invocations,
       any(t.LlmCount) AS llm_calls, any(t.ToolCount) AS tool_calls,
       any(t.InputTokens) AS input_tokens, any(t.OutputTokens) AS output_tokens,
       any(t.Models) AS models, any(t.ErrorCount) AS error_count,
       if(countIf(s.response_id != '') = 0, NULL, sum(s.spend)) AS spend
FROM (
    SELECT TeamId, TraceId, min(StartTs) AS StartTs, max(EndTs) AS EndTs,
           any(ServiceName) AS ServiceName, anyLastIf(a.RootName, a.RootName != '') AS RootName,
           anyLastIf(a.RootInput, a.RootName != '') AS RootInput,
           anyLastIf(a.RootStatus, a.RootName != '') AS RootStatus,
           sum(SpanCount) AS SpanCount, sum(AgentCount) AS AgentCount, sum(LlmCount) AS LlmCount,
           sum(ToolCount) AS ToolCount, sum(ErrorCount) AS ErrorCount, sum(InputTokens) AS InputTokens,
           sum(OutputTokens) AS OutputTokens, groupUniqArrayArray(Models) AS Models,
           groupUniqArrayArray(AgentNames) AS AgentNames, groupArrayArray(RequestIds) AS RequestIds
    FROM agent_traces AS a
    WHERE (empty({team_ids:Array(String)}) OR TeamId IN {team_ids:Array(String)})
      AND ({api_key_hash:String} = '' OR TraceId IN (
          SELECT TraceId FROM otel_traces WHERE (empty({team_ids:Array(String)}) OR TeamId IN {team_ids:Array(String)}) AND ({api_key_hash:String} = '' OR ApiKeyHash = {api_key_hash:String})))
    GROUP BY TeamId, TraceId
    HAVING StartTs >= fromUnixTimestamp64Milli({start_ms:Int64})
       AND StartTs < fromUnixTimestamp64Milli({end_ms:Int64})
       AND (({cursor_ms:Int64} = 0) OR (toUnixTimestamp64Milli(StartTs), TraceId)
            < ({cursor_ms:Int64}, {cursor_trace_id:String}))
    ORDER BY StartTs DESC, TraceId DESC
    LIMIT {limit:UInt32}
) AS t
LEFT ARRAY JOIN t.RequestIds AS request_id
LEFT JOIN (
    SELECT response_id, any(spend) AS spend FROM spend_logs FINAL WHERE (empty({team_ids:Array(String)}) OR team_id IN {team_ids:Array(String)}) AND ({api_key_hash:String} = '' OR api_key = {api_key_hash:String}) GROUP BY response_id
) AS s ON s.response_id = request_id
GROUP BY t.TraceId
ORDER BY start_ms DESC, t.TraceId DESC
