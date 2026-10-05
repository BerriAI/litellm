SELECT o.SpanId AS span_id, o.Input AS input,
       if(o.Output = '' AND o.ObservationType = 'agent', answer.output, o.Output) AS output,
       o.SpanAttributes AS attributes
FROM otel_traces AS o
LEFT JOIN (
    SELECT TeamId, ApiKeyHash, ParentSpanId AS parent_span_id, argMax(Output, Timestamp) AS output
    FROM otel_traces
    WHERE TraceId = {trace_id:String} AND ParentSpanId = {span_id:String}
      AND ObservationType = 'llm' AND Output != ''
      AND ({all_teams:UInt8} = 1
           OR ({user_id:String} != '' AND UserId = {user_id:String})
           OR has({team_ids:Array(String)}, TeamId))
      AND ({trace_ref:String} = '' OR
           hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))) = {trace_ref:String})
    GROUP BY TeamId, ApiKeyHash, ParentSpanId
) AS answer ON answer.parent_span_id = o.SpanId
    AND answer.TeamId = o.TeamId AND answer.ApiKeyHash = o.ApiKeyHash
WHERE o.TraceId = {trace_id:String} AND o.SpanId = {span_id:String}
  AND ({all_teams:UInt8} = 1
       OR ({user_id:String} != '' AND o.UserId = {user_id:String})
       OR has({team_ids:Array(String)}, o.TeamId))
  AND ({trace_ref:String} = '' OR
       hex(SHA256(concat(o.TeamId, char(0), o.ApiKeyHash, char(0), o.TraceId))) = {trace_ref:String})
LIMIT 1
