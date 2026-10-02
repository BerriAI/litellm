SELECT SpanId AS span_id, Input AS input, Output AS output, SpanAttributes AS attributes
FROM otel_traces
WHERE TraceId = {trace_id:String} AND SpanId = {span_id:String}
  AND (empty({team_ids:Array(String)}) OR TeamId IN {team_ids:Array(String)})
  AND ({api_key_hash:String} = '' OR ApiKeyHash = {api_key_hash:String})
  AND ({trace_ref:String} = '' OR
       hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))) = {trace_ref:String})
LIMIT 1
