INSERT INTO "LiteLLM_SpendLogs" (
    request_id, call_type, api_key, spend, total_tokens, prompt_tokens,
    completion_tokens, "startTime", "endTime", "completionStartTime",
    request_duration_ms, model, model_id, model_group, custom_llm_provider, api_base,
    "user", metadata, cache_hit, cache_key, request_tags, team_id, organization_id,
    end_user, requester_ip_address, messages, response, proxy_server_request,
    session_id, status, litellm_call_id
)
SELECT l.request_id, l.call_type, l.api_key, l.spend, l.total_tokens, l.prompt_tokens,
       l.completion_tokens, l.start_time::timestamp(3), l.end_time::timestamp(3),
       l.completion_start_time::timestamp(3), l.request_duration_ms, l.model,
       l.model_id, l.model_group, l.custom_llm_provider, l.api_base, l.user_id,
       l.metadata, l.cache_hit, l.cache_key, '[]'::jsonb, l.team_id,
       l.organization_id, l.end_user, l.requester_ip_address, '{}'::jsonb,
       '{}'::jsonb, '{}'::jsonb, l.session_id, l.status, l.litellm_call_id
FROM UNNEST($1::text[], $2::text[], $3::text[], $4::float8[], $5::int4[], $6::int4[],
            $7::int4[], $8::timestamp[], $9::timestamp[], $10::timestamp[], $11::int4[], $12::text[],
            $13::text[], $14::text[], $15::text[], $16::text[], $17::text[],
            $18::jsonb[], $19::text[], $20::text[], $21::text[], $22::text[],
            $23::text[], $24::text[], $25::text[], $26::text[], $27::text[])
     AS l(request_id, call_type, api_key, spend, total_tokens, prompt_tokens,
          completion_tokens, start_time, end_time, completion_start_time,
          request_duration_ms, model, model_id, model_group, custom_llm_provider,
          api_base, user_id, metadata, cache_hit, cache_key, team_id, organization_id,
          end_user, requester_ip_address, session_id, status, litellm_call_id)
ON CONFLICT DO NOTHING
