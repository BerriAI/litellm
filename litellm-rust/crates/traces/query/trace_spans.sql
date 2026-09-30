SELECT o.SpanId AS span_id, o.ParentSpanId AS parent_span_id, o.SpanName AS name,
       o.ObservationType AS type, o.AgentName AS agent, o.StatusCode AS status,
       o.StatusMessage AS status_message,
       toUnixTimestamp64Nano(o.Timestamp) AS start_ns, o.Duration AS duration_ns,
       o.ServiceName AS service, o.InputPreview AS input_preview, o.Model AS model,
       o.InputTokens AS input_tokens, o.OutputTokens AS output_tokens,
       o.LiteLLMRequestId AS litellm_request_id,
       s.request_id AS s_request_id, s.model AS s_model, s.model_group AS s_model_group,
       s.custom_llm_provider AS s_provider, s.api_base AS s_api_base, s.key_alias AS s_key_alias,
       s.team_alias AS s_team_alias, s.spend AS s_spend, s.prompt_tokens AS s_prompt_tokens,
       s.completion_tokens AS s_completion_tokens, s.cache_read_tokens AS s_cache_read_tokens,
       s.cache_write_tokens AS s_cache_write_tokens,
       toUnixTimestamp64Milli(s.start_time) AS s_start_ms, toUnixTimestamp64Milli(s.end_time) AS s_end_ms,
       if(isNull(s.completion_start_time), 0, toUnixTimestamp64Milli(assumeNotNull(s.completion_start_time)))
           AS s_ttft_start_ms,
       s.status AS s_status
FROM otel_traces AS o
LEFT JOIN (
    SELECT * FROM spend_logs FINAL
    WHERE response_id IN (
        SELECT LiteLLMRequestId FROM otel_traces
        WHERE TraceId = {trace_id:String} AND LiteLLMRequestId != '' AND (empty({team_ids:Array(String)}) OR TeamId IN {team_ids:Array(String)}) AND ({api_key_hash:String} = '' OR ApiKeyHash = {api_key_hash:String}))
      AND (empty({team_ids:Array(String)}) OR team_id IN {team_ids:Array(String)}) AND ({api_key_hash:String} = '' OR api_key = {api_key_hash:String})
) AS s ON s.response_id = o.LiteLLMRequestId
WHERE o.TraceId = {trace_id:String} AND (empty({team_ids:Array(String)}) OR TeamId IN {team_ids:Array(String)}) AND ({api_key_hash:String} = '' OR ApiKeyHash = {api_key_hash:String})
ORDER BY o.Timestamp, abs(toUnixTimestamp64Milli(s.start_time) - toUnixTimestamp64Milli(o.Timestamp))
LIMIT 1 BY o.SpanId
