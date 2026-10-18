INSERT INTO "LiteLLM_SpendLogs"
SELECT (jsonb_populate_record(NULL::"LiteLLM_SpendLogs", to_jsonb(template) || jsonb_build_object(
    'request_id', 'perf-ui-' || g,
    'litellm_call_id', md5('perf-ui-call-' || g),
    'api_key', %(api_key)s::text,
    'team_id', 'perf-team-0',
    'user', 'perf-user-' || g %% 200,
    'end_user', 'perf-end-user-' || g %% 200,
    'model', 'perf-chat', 'model_id', 'perf-model-0', 'model_group', 'perf-chat',
    'custom_llm_provider', 'openai',
    'startTime', %(anchor)s::timestamp - g * interval '10 seconds',
    'endTime', %(anchor)s::timestamp - g * interval '10 seconds' + interval '1 second',
    'completionStartTime', %(anchor)s::timestamp - g * interval '10 seconds' + interval '100 milliseconds',
    'session_id', CASE WHEN g %% 2 = 0 THEN 'perf-ui-session' ELSE 'perf-ui-session-' || g END,
    'status', CASE WHEN g %% 10 = 0 THEN 'failure' ELSE 'success' END,
    'cache_hit', CASE WHEN g %% 10 = 1 THEN 'True' ELSE 'False' END,
    'spend', 0.01, 'prompt_tokens', 1000, 'completion_tokens', 100, 'total_tokens', 1100,
    'messages', '{}', 'response', '{}', 'proxy_server_request', '{}',
    'metadata', jsonb_build_object(
        'used_client_oauth_token', false,
        'user_api_key_alias', 'perf-key-0', 'user_api_key_team_alias', 'perf-team-0',
        'usage_object', jsonb_build_object('prompt_tokens', 1000, 'completion_tokens', 100,
            'cache_read_input_tokens', 500, 'cache_creation_input_tokens', 100),
        %(gateway_marker)s::text, 'perf-model-0',
        'error_information', CASE WHEN g %% 10 = 0 THEN
            jsonb_build_object('error_code', '429', 'error_message', 'Synthetic upstream rate limit') ELSE '{}'::jsonb END,
        'guardrail_information', jsonb_build_array(jsonb_build_object(
            'guardrail_name', 'perf-ui-guardrail', 'guardrail_status', 'success', 'duration', 0.01))
    )
))).*
FROM "LiteLLM_SpendLogs" template CROSS JOIN generate_series(1, 320) g
WHERE template.request_id = 'perf-seed-1'
ON CONFLICT (request_id) DO NOTHING;

INSERT INTO "LiteLLM_GuardrailsTable" (guardrail_id, guardrail_name, litellm_params, updated_at)
VALUES ('perf-ui-guardrail', 'perf-ui-guardrail',
    '{"guardrail":"generic_guardrail_api","mode":"pre_call","api_base":"http://127.0.0.1:1"}', now())
ON CONFLICT (guardrail_id) DO NOTHING;

INSERT INTO "LiteLLM_SpendLogGuardrailIndex" (request_id, guardrail_id, start_time)
SELECT request_id, 'perf-ui-guardrail', "startTime" FROM "LiteLLM_SpendLogs"
WHERE request_id = ANY(%(request_ids)s)
ON CONFLICT DO NOTHING;

INSERT INTO "LiteLLM_SpendLogToolIndex" (request_id, tool_name, start_time)
SELECT request_id, 'perf-ui-tool', "startTime" FROM "LiteLLM_SpendLogs"
WHERE request_id = ANY(%(request_ids)s)
ON CONFLICT DO NOTHING;
