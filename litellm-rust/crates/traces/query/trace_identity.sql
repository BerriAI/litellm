SELECT hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))) AS trace_ref
FROM otel_traces
WHERE TraceId = {trace_id:String}
  AND (empty({team_ids:Array(String)}) OR TeamId IN {team_ids:Array(String)})
  AND ({api_key_hash:String} = '' OR ApiKeyHash = {api_key_hash:String})
GROUP BY TeamId, ApiKeyHash, TraceId
LIMIT 2
