WITH
    coalesce(nullIf(SpanAttributes['query_source_safe'], ''), SpanAttributes['query_source']) AS native_source,
    ((Framework = 'claude-code' OR (Framework = 'claude-agent-sdk' AND
      (startsWith(SpanName, 'claude_code.') OR SpanAttributes['agent_id'] != '' OR native_source != '')))
      AND (SpanAttributes['session.id'] != '' OR SpanAttributes['agent_id'] != '')) AS native,
    if(SpanAttributes['session.id'] = '', TraceId, SpanAttributes['session.id']) AS native_session,
    (native AND SpanAttributes['agent_id'] != '') AS native_child,
    (native AND NOT native_child AND (SpanName = 'claude_code.interaction' OR
        native_source IN
        ('repl_main_thread', 'sdk', 'sdk_main_thread', 'generate_session_title', 'prompt_suggestion'))) AS native_root,
    page AS (
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
SELECT page.* EXCEPT (trace_start, trace_end, agent_invocations),
       if(identities.native_count > 0, identities.resolved_invocations, page.agent_invocations) AS agent_invocations,
       identities.agent_names AS agent_names, identities.agent_count AS agent_count,
       identities.frameworks AS frameworks
FROM page
LEFT JOIN (
    SELECT TeamId, ApiKeyHash, TraceId,
           arraySort(groupUniqArrayArray(agent_names)) AS agent_names,
           arraySort(groupUniqArrayArray(frameworks)) AS frameworks,
           countIf(native_actor) AS native_count,
           native_count + uniqExactIf(legacy_name, legacy_invocations > 0) AS agent_count,
           sum(if(native_actor, greatest(1, native_interactions), legacy_invocations)) AS resolved_invocations
    FROM (
        SELECT TeamId, ApiKeyHash, TraceId,
               native_root OR native_child AS native_actor,
               (native_session, if(native_child, concat('agent:', SpanAttributes['agent_id']), 'root')) AS actor_identity,
               if(native_actor, '', if(AgentName = '', SpanName, AgentName)) AS legacy_name,
               groupUniqArrayIf(AgentName, AgentName != '') AS agent_names,
               groupUniqArrayIf(toString(Framework), Framework != '') AS frameworks,
               uniqExactIf(SpanId, native_actor AND SpanName = 'claude_code.interaction') AS native_interactions,
               uniqExactIf(SpanId, ObservationType = 'agent' AND NOT native) AS legacy_invocations
        FROM otel_traces
        WHERE Timestamp >= (SELECT min(trace_start) FROM page)
          AND Timestamp <= (SELECT max(trace_end) FROM page)
          AND TraceId IN (SELECT trace_id FROM page)
          AND (TeamId, ApiKeyHash, TraceId) IN (SELECT team_id, api_key_hash, trace_id FROM page)
        GROUP BY TeamId, ApiKeyHash, TraceId, native_actor, actor_identity, legacy_name
    )
    GROUP BY TeamId, ApiKeyHash, TraceId
) AS identities
ON page.team_id = identities.TeamId AND page.api_key_hash = identities.ApiKeyHash
   AND page.trace_id = identities.TraceId
ORDER BY page.start_ms DESC, page.trace_ref DESC
