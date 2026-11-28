INSERT INTO "LiteLLM_DailyModelUsage" AS stored (date, model_group, model, custom_llm_provider, task_type, spend, prompt_tokens, completion_tokens, request_count, successful_requests, failed_requests, total_response_time_ms, timed_requests, updated_at)
SELECT date, model_group, model, custom_llm_provider, task_type, spend, prompt_tokens, completion_tokens, request_count, successful_requests, failed_requests, total_response_time_ms, timed_requests, (NOW() AT TIME ZONE 'UTC')
FROM jsonb_to_recordset($1::jsonb) AS input(
    date text,
    model_group text,
    model text,
    custom_llm_provider text,
    task_type text,
    spend double precision,
    prompt_tokens bigint,
    completion_tokens bigint,
    request_count bigint,
    successful_requests bigint,
    failed_requests bigint,
    total_response_time_ms bigint,
    timed_requests bigint
)
ORDER BY date COLLATE "C", model_group COLLATE "C", model COLLATE "C", custom_llm_provider COLLATE "C", task_type COLLATE "C"
ON CONFLICT (date, model_group, model, custom_llm_provider, task_type) DO UPDATE SET
    spend = stored.spend + EXCLUDED.spend,
    prompt_tokens = stored.prompt_tokens + EXCLUDED.prompt_tokens,
    completion_tokens = stored.completion_tokens + EXCLUDED.completion_tokens,
    request_count = stored.request_count + EXCLUDED.request_count,
    successful_requests = stored.successful_requests + EXCLUDED.successful_requests,
    failed_requests = stored.failed_requests + EXCLUDED.failed_requests,
    total_response_time_ms = stored.total_response_time_ms + EXCLUDED.total_response_time_ms,
    timed_requests = stored.timed_requests + EXCLUDED.timed_requests,
    updated_at = EXCLUDED.updated_at
