SELECT SpanId AS span_id, Input AS input, Output AS output, SpanAttributes AS attributes
FROM otel_traces
WHERE TraceId = {trace_id:String} AND SpanId = {span_id:String} AND (empty({team_ids:Array(String)}) OR TeamId IN {team_ids:Array(String)}) AND ({api_key_hash:String} = '' OR ApiKeyHash = {api_key_hash:String})
LIMIT 1
