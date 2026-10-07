WITH runs AS (
    SELECT TeamId, ApiKeyHash, TraceId, min(StartTs) AS trace_start, max(EndTs) AS trace_end
    FROM agent_traces_by_key
    WHERE ({all_teams:UInt8} = 1
           OR ({user_id:String} != '' AND UserIds = [{user_id:String}])
           OR has({team_ids:Array(String)}, TeamId))
    GROUP BY TeamId, ApiKeyHash, TraceId
    HAVING trace_start >= fromUnixTimestamp64Milli({start_ms:Int64})
       AND trace_start < fromUnixTimestamp64Milli({end_ms:Int64})
)
SELECT DISTINCT arrayJoin(if(empty(names), [first_service], names)) AS agent_name
FROM (
    -- Keep in sync with list_traces.sql: the run's agents as resolve_trace names them.
    SELECT if(countIf(AgentName != '') = 0,
              groupUniqArrayIf(SpanName, ObservationType = 'agent'),
              arrayConcat(groupUniqArrayIf(AgentName, AgentName != ''),
                          groupUniqArrayIf(SpanName, ObservationType = 'agent' AND AgentName = ''
                                                     AND NOT WrapperCandidate))) AS names,
           argMin(ServiceName, Timestamp) AS first_service
    FROM otel_traces
    WHERE Timestamp >= (SELECT min(trace_start) FROM runs)
      AND Timestamp <= (SELECT max(trace_end) FROM runs)
      AND (TeamId, ApiKeyHash, TraceId) IN (SELECT TeamId, ApiKeyHash, TraceId FROM runs)
    GROUP BY TeamId, ApiKeyHash, TraceId
)
WHERE agent_name != ''
ORDER BY agent_name
LIMIT {limit:UInt32}
