SELECT o.TraceId AS trace_id, o.SpanId AS span_id, o.ParentSpanId AS parent_span_id, o.SpanName AS name,
       o.ObservationType AS type, toUInt8(o.WrapperCandidate) AS wrapper_candidate, o.AgentName AS agent,
       o.Framework AS framework, o.StatusCode AS status,
       substringUTF8(o.StatusMessage, 1, 128) AS status_message,
       lengthUTF8(o.StatusMessage) > 128 AS error_truncated,
       toUnixTimestamp64Nano(o.Timestamp) AS start_ns, o.Duration AS duration_ns,
       o.ServiceName AS service, o.InputPreview AS input_preview, o.Model AS model,
       o.InputTokens AS input_tokens, o.OutputTokens AS output_tokens,
       o.LiteLLMRequestId AS litellm_request_id,
       o.CallKeys AS call_keys, o.CallEvidence AS call_evidence,
       -- Rows written before ToolCallId keep the call id only in their attributes.
       if(o.ToolCallId != '' OR o.ObservationType != 'tool', o.ToolCallId,
          coalesce(nullIf(o.SpanAttributes['gen_ai.tool.call.id'], ''), nullIf(o.SpanAttributes['tool.id'], ''), ''))
          AS tool_call_id,
       o.UserId AS user_id, o.TeamId AS team_id, o.ApiKeyHash AS api_key_hash
FROM otel_traces AS o
WHERE o.TraceId = {trace_id:String}
  AND ({all_teams:UInt8} = 1
       OR ({user_id:String} != '' AND o.UserId = {user_id:String})
       OR has({team_ids:Array(String)}, o.TeamId))
  AND ({trace_ref:String} = '' OR
       hex(SHA256(concat(o.TeamId, char(0), o.ApiKeyHash, char(0), o.TraceId))) = {trace_ref:String})
ORDER BY o.Timestamp, o.EngineReceivedMs, o.StatusMessage
LIMIT 1 BY o.SpanId
