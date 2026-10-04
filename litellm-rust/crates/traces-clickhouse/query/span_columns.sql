SELECT TraceId AS trace_id, SpanId AS span_id, ParentSpanId AS parent_span_id, SpanName AS name,
       ObservationType AS type, toUInt8(WrapperCandidate) AS wrapper_candidate, AgentName AS agent,
       Framework AS framework, StatusCode AS status,
       substringUTF8(StatusMessage, 1, 128) AS status_message,
       lengthUTF8(StatusMessage) > 128 AS error_truncated,
       toUnixTimestamp64Nano(Timestamp) AS start_ns, Duration AS duration_ns,
       ServiceName AS service, InputPreview AS input_preview, Model AS model,
       InputTokens AS input_tokens, OutputTokens AS output_tokens,
       LiteLLMRequestId AS litellm_request_id,
       CallKeys AS call_keys, CallEvidence AS call_evidence,
       -- Rows written before ToolCallId keep the call id only in their attributes.
       if(ToolCallId != '' OR ObservationType != 'tool', ToolCallId,
          coalesce(nullIf(SpanAttributes['gen_ai.tool.call.id'], ''), nullIf(SpanAttributes['tool.id'], ''), ''))
          AS tool_call_id,
       UserId AS user_id, TeamId AS team_id, ApiKeyHash AS api_key_hash
FROM owned_spans
