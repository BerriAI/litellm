SELECT hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))) AS trace_ref
FROM otel_traces
WHERE TraceId = {trace_id:String}
  AND ({all_teams:UInt8} = 1
       OR ({user_id:String} != '' AND UserId = {user_id:String})
       OR has({team_ids:Array(String)}, TeamId)
       OR ({api_key_hash:String} != '' AND ApiKeyHash = {api_key_hash:String}))
GROUP BY TeamId, ApiKeyHash, TraceId
LIMIT 2
