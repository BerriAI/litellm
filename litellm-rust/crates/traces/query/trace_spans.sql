SELECT o.SpanId AS span_id, o.ParentSpanId AS parent_span_id, o.SpanName AS name,
       o.ObservationType AS type, o.AgentName AS agent, o.StatusCode AS status,
       substringUTF8(o.StatusMessage, 1, 128) AS status_message,
       lengthUTF8(o.StatusMessage) > 128 AS error_truncated,
       toUnixTimestamp64Nano(o.Timestamp) AS start_ns, o.Duration AS duration_ns,
       o.ServiceName AS service, o.InputPreview AS input_preview, o.Model AS model,
       o.InputTokens AS input_tokens, o.OutputTokens AS output_tokens,
       o.LiteLLMRequestId AS litellm_request_id,
       o.TeamId AS team_id, o.ApiKeyHash AS api_key_hash
FROM otel_traces AS o
WHERE o.TraceId = {trace_id:String}
  AND (empty({team_ids:Array(String)}) OR o.TeamId IN {team_ids:Array(String)})
  AND ({api_key_hash:String} = '' OR o.ApiKeyHash = {api_key_hash:String})
  AND ({trace_ref:String} = '' OR
       hex(SHA256(concat(o.TeamId, char(0), o.ApiKeyHash, char(0), o.TraceId))) = {trace_ref:String})
ORDER BY o.Timestamp, o.EngineReceivedMs, o.StatusMessage
LIMIT 1 BY o.SpanId
