SELECT {span_id:String} AS span_id, span.1 AS input,
       if(span.2 = '' AND span.4 = 'agent', answer, span.2) AS output,
       span.3 AS attributes
FROM (
    SELECT TeamId, ApiKeyHash,
           argMinIf((Input, Output, SpanAttributes, ObservationType), (Timestamp, EngineReceivedMs, StatusMessage),
                    SpanId = {span_id:String}) AS span,
           argMaxIf(Output, Timestamp,
                    ParentSpanId = {span_id:String} AND ObservationType = 'llm' AND Output != '') AS answer
    FROM otel_traces
    PREWHERE TraceId = {trace_id:String} AND (SpanId = {span_id:String} OR ParentSpanId = {span_id:String})
    WHERE {all_teams:UInt8} = 1
          OR ({user_id:String} != '' AND UserId = {user_id:String})
          OR has({team_ids:Array(String)}, TeamId)
    GROUP BY TeamId, ApiKeyHash
    HAVING countIf(SpanId = {span_id:String}) > 0
       AND ({trace_ref:String} = '' OR
            hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), {trace_id:String}))) = {trace_ref:String})
)
LIMIT 1
