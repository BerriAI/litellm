WITH candidates AS (
SELECT TeamId, ApiKeyHash, TraceId
FROM agent_traces_by_key
WHERE StartTs >= fromUnixTimestamp64Milli({start_ms:Int64})
  AND StartTs < fromUnixTimestamp64Milli({end_ms:Int64})
  AND ({all_teams:UInt8} = 1
       OR ({user_id:String} != '' AND UserIds = [{user_id:String}])
       OR has({team_ids:Array(String)}, TeamId))
),
runs AS (
SELECT TeamId, ApiKeyHash, TraceId,
       toUnixTimestamp64Milli(min(StartTs)) AS start_ms,
       min(StartTs) AS trace_start, max(EndTs) AS trace_end,
       sum(ErrorCount) > 0 AS failed
FROM agent_traces_by_key
WHERE (TeamId, ApiKeyHash, TraceId) IN candidates
  AND ({all_teams:UInt8} = 1
       OR ({user_id:String} != '' AND UserIds = [{user_id:String}])
       OR has({team_ids:Array(String)}, TeamId))
GROUP BY TeamId, ApiKeyHash, TraceId
HAVING min(StartTs) >= fromUnixTimestamp64Milli({start_ms:Int64})
   AND min(StartTs) < fromUnixTimestamp64Milli({end_ms:Int64})
),
named AS (
SELECT DISTINCT o.TeamId AS TeamId, o.ApiKeyHash AS ApiKeyHash, o.TraceId AS TraceId,
       o.AgentName AS agent_name, toString(o.Framework) AS framework
FROM otel_traces AS o
WHERE o.AgentName != ''
  AND o.Timestamp >= (SELECT min(trace_start) FROM runs)
  AND o.Timestamp <= (SELECT max(trace_end) FROM runs)
  AND (o.TeamId, o.ApiKeyHash, o.TraceId) IN (SELECT TeamId, ApiKeyHash, TraceId FROM runs)
)
SELECT named.agent_name AS agent_name,
       uniqExact(named.TeamId, named.ApiKeyHash, named.TraceId) AS runs,
       uniqExactIf((named.TeamId, named.ApiKeyHash, named.TraceId), runs.failed) AS failed_runs,
       max(runs.start_ms) AS last_seen_ms,
       arraySort(groupUniqArrayIf(named.framework, named.framework != '')) AS frameworks
FROM named
INNER JOIN runs ON named.TeamId = runs.TeamId AND named.ApiKeyHash = runs.ApiKeyHash
               AND named.TraceId = runs.TraceId
GROUP BY named.agent_name
ORDER BY last_seen_ms DESC, agent_name
LIMIT {limit:UInt32}
