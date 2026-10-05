SELECT TraceId AS trace_id,
       hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))) AS trace_ref,
       if(length(groupUniqArrayArray(UserIds)) = 1, arrayElement(groupUniqArrayArray(UserIds), 1), '') AS user_id, TeamId AS team_id, ApiKeyHash AS api_key_hash,
       ifNull(any(RootName), '') AS name, any(ServiceName) AS service,
       ifNull(any(RootInput), '') AS input_preview, ifNull(any(RootStatus), '') AS status,
       toUnixTimestamp64Milli(min(StartTs)) AS start_ms,
       if(any(metrics.span_count) > 0,
          toUInt64(greatest(toUnixTimestamp64Nano(any(metrics.end_ts)) - toUnixTimestamp64Nano(any(metrics.start_ts)), 0)),
          toUInt64(greatest(toUnixTimestamp64Nano(max(EndTs)) - toUnixTimestamp64Nano(min(StartTs)), 0))) AS duration_ns,
       if(any(metrics.span_count) > 0, any(metrics.span_count), sum(SpanCount)) AS span_count,
       sum(AgentCount) AS agent_invocations,
       sum(LlmCount) AS llm_calls, sum(ToolCount) AS tool_calls,
       sum(InputTokens) AS input_tokens, sum(OutputTokens) AS output_tokens,
       groupUniqArrayArray(Models) AS models,
       if(any(metrics.span_count) > 0, any(metrics.error_count), sum(ErrorCount)) AS error_count,
       arraySort(groupUniqArrayArray(AgentNames)) AS search_agents,
       if(error_count > 0, 'error', 'ok') AS search_status,
       length(groupUniqArrayArray(AgentIdentities)) AS agent_count,
       arraySort(groupUniqArrayArray(Frameworks)) AS frameworks
FROM owned_runs
LEFT JOIN (
    SELECT TeamId, ApiKeyHash, TraceId,
           min(Timestamp) AS start_ts,
           max(Timestamp + toIntervalNanosecond(Duration)) AS end_ts,
           count() AS span_count,
           countIf(StatusCode = 'STATUS_CODE_ERROR') AS error_count
    FROM (
        SELECT TeamId, ApiKeyHash, TraceId, SpanId, Timestamp, Duration, StatusCode
        FROM owned_spans
        WHERE (TeamId, ApiKeyHash, TraceId) IN (
              SELECT TeamId, ApiKeyHash, TraceId
              FROM owned_runs
              WHERE {trace_id:String} = '' OR TraceId = {trace_id:String}
              GROUP BY TeamId, ApiKeyHash, TraceId
              HAVING {trace_id:String} != '' OR (
                  min(StartTs) >= fromUnixTimestamp64Milli({start_ms:Int64})
                  AND min(StartTs) < fromUnixTimestamp64Milli({end_ms:Int64}))
          )
          AND ({trace_id:String} != '' OR Timestamp >= fromUnixTimestamp64Milli({start_ms:Int64}))
        ORDER BY Timestamp, EngineReceivedMs, StatusMessage, Duration, StatusCode
        LIMIT 1 BY TeamId, ApiKeyHash, TraceId, SpanId
    )
    GROUP BY TeamId, ApiKeyHash, TraceId
) AS metrics USING (TeamId, ApiKeyHash, TraceId)
LEFT JOIN (
    SELECT TeamId, ApiKeyHash, TraceId,
           groupUniqArrayArray(arrayFilter(i -> (mapContains(ResourceAttributes, {attribute_keys:Array(String)}[i])
                                                AND ResourceAttributes[{attribute_keys:Array(String)}[i]] ILIKE {attribute_patterns:Array(String)}[i])
                                             OR (mapContains(SpanAttributes, {attribute_keys:Array(String)}[i])
                                                AND SpanAttributes[{attribute_keys:Array(String)}[i]] ILIKE {attribute_patterns:Array(String)}[i]),
                                       arrayEnumerate({attribute_keys:Array(String)}))) AS matched_attributes
    FROM owned_spans
    WHERE notEmpty({attribute_keys:Array(String)})
      AND Timestamp >= fromUnixTimestamp64Milli({start_ms:Int64})
    GROUP BY TeamId, ApiKeyHash, TraceId
) AS attributes USING (TeamId, ApiKeyHash, TraceId)
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
                f = 'service', service ILIKE p,
                f = 'team', team_id ILIKE p,
                false),
              {filter_fields:Array(String)}, {filter_patterns:Array(String)}, {filter_modes:Array(String)})
        AND arrayAll((i, m) -> (m = 'exclude') != has(any(matched_attributes), i),
              arrayEnumerate({attribute_keys:Array(String)}), {attribute_modes:Array(String)})
        AND (empty({trace_refs:Array(String)}) OR trace_ref IN {trace_refs:Array(String)}))
