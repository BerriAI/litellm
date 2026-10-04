page AS (
SELECT * EXCEPT (search_agents, search_status)
FROM runs
WHERE {has_cursor:UInt8} = 0 OR (start_ms, trace_ref) < ({cursor_ms:Int64}, {cursor_ref:String})
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
    FROM owned_spans
    WHERE Timestamp >= (SELECT min(trace_start) FROM page)
      AND Timestamp <= (SELECT max(trace_end) FROM page)
      AND TraceId IN (SELECT trace_id FROM page)
      AND (TeamId, ApiKeyHash, TraceId) IN (SELECT team_id, api_key_hash, trace_id FROM page)
    GROUP BY TeamId, ApiKeyHash, TraceId
) AS identities
ON page.team_id = identities.TeamId AND page.api_key_hash = identities.ApiKeyHash
   AND page.trace_id = identities.TraceId
ORDER BY page.start_ms DESC, page.trace_ref DESC
