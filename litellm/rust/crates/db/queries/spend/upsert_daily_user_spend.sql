INSERT INTO "LiteLLM_DailyUserSpend" (
    id, user_id, date, api_key, model, custom_llm_provider,
    mcp_namespaced_tool_name, endpoint, model_group, prompt_tokens,
    completion_tokens, api_requests, successful_requests, failed_requests,
    cache_read_input_tokens, cache_creation_input_tokens, total_response_time_ms,
    timed_requests, spend, updated_at
)
SELECT gen_random_uuid()::text, r.entity_id, r.date, r.api_key, r.model,
       r.custom_llm_provider, r.mcp_namespaced_tool_name, r.endpoint, r.model_group, r.prompt_tokens,
       r.completion_tokens, r.api_requests, r.successful_requests,
       r.failed_requests, r.cache_read, r.cache_creation, r.response_time,
       r.timed_requests, r.spend, (NOW() AT TIME ZONE 'UTC')
FROM UNNEST(
    $1::text[], $2::text[], $3::text[], $4::text[], $5::text[],
    $6::text[], $7::text[], $8::int8[], $9::int8[], $10::int8[],
    $11::int8[], $12::int8[], $13::int8[], $14::int8[], $15::int8[],
    $16::int8[], $17::float8[], $18::text[]
) AS r(
    entity_id, date, api_key, model, custom_llm_provider, endpoint,
    model_group, prompt_tokens, completion_tokens, api_requests,
    successful_requests, failed_requests, cache_read, cache_creation,
    response_time, timed_requests, spend, mcp_namespaced_tool_name
)
ORDER BY r.date, r.entity_id, r.api_key, r.model, r.custom_llm_provider,
         r.mcp_namespaced_tool_name, r.endpoint
ON CONFLICT (
    user_id, date, api_key, model, custom_llm_provider,
    mcp_namespaced_tool_name, endpoint
) DO UPDATE SET
    prompt_tokens = "LiteLLM_DailyUserSpend".prompt_tokens + EXCLUDED.prompt_tokens,
    completion_tokens = "LiteLLM_DailyUserSpend".completion_tokens + EXCLUDED.completion_tokens,
    api_requests = "LiteLLM_DailyUserSpend".api_requests + EXCLUDED.api_requests,
    successful_requests = "LiteLLM_DailyUserSpend".successful_requests + EXCLUDED.successful_requests,
    failed_requests = "LiteLLM_DailyUserSpend".failed_requests + EXCLUDED.failed_requests,
    cache_read_input_tokens = "LiteLLM_DailyUserSpend".cache_read_input_tokens + EXCLUDED.cache_read_input_tokens,
    cache_creation_input_tokens = "LiteLLM_DailyUserSpend".cache_creation_input_tokens + EXCLUDED.cache_creation_input_tokens,
    total_response_time_ms = "LiteLLM_DailyUserSpend".total_response_time_ms + EXCLUDED.total_response_time_ms,
    timed_requests = "LiteLLM_DailyUserSpend".timed_requests + EXCLUDED.timed_requests,
    spend = "LiteLLM_DailyUserSpend".spend + EXCLUDED.spend,
    updated_at = (NOW() AT TIME ZONE 'UTC')
