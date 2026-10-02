SELECT o.SpanId AS span_id, o.Input AS input,
       if(o.Output = '' AND o.ObservationType = 'agent', answer.output, o.Output) AS output,
       o.SpanAttributes AS attributes
FROM otel_traces AS o
LEFT JOIN (
    SELECT ParentSpanId AS parent_span_id, argMax(Output, Timestamp) AS output
    FROM otel_traces
    WHERE TraceId = {trace_id:String} AND ParentSpanId = {span_id:String}
      AND ObservationType = 'llm' AND Output != ''
      AND (empty({team_ids:Array(String)}) OR TeamId IN {team_ids:Array(String)})
      AND ({api_key_hash:String} = '' OR ApiKeyHash = {api_key_hash:String})
    GROUP BY ParentSpanId
) AS answer ON answer.parent_span_id = o.SpanId
WHERE o.TraceId = {trace_id:String} AND o.SpanId = {span_id:String}
  AND (empty({team_ids:Array(String)}) OR o.TeamId IN {team_ids:Array(String)})
  AND ({api_key_hash:String} = '' OR o.ApiKeyHash = {api_key_hash:String})
  AND ({trace_ref:String} = '' OR
       hex(SHA256(concat(o.TeamId, char(0), o.ApiKeyHash, char(0), o.TraceId))) = {trace_ref:String})
LIMIT 1
