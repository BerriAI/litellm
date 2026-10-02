SELECT TeamId AS team, ApiKeyHash AS api_key, TraceId AS trace_id,
       ifNull(any(RootName), '') AS name,
       toUInt32(sum(SpanCount)) AS spans,
       toUInt32(sum(LlmCount)) AS llm_calls,
       toUInt32(sum(ErrorCount)) AS errors,
       toUInt32(sum(InputTokens)) AS input_tokens,
       toUInt32(sum(OutputTokens)) AS output_tokens
FROM agent_traces_by_key
GROUP BY TeamId, ApiKeyHash, TraceId
ORDER BY team, api_key, trace_id
