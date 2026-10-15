WITH hashes AS (
    SELECT g,
        ('x' || substr(md5('key:' || ((g - 1) / 8)::text), 1, 8))::bit(32)::bigint AS key_hash,
        ('x' || substr(md5('time:' || ((g - 1) / 8)::text), 1, 8))::bit(32)::bigint AS time_hash,
        ('x' || substr(md5('tokens:' || g::text), 1, 8))::bit(32)::bigint AS token_hash
    FROM generate_series(%(first)s::bigint, %(last)s::bigint) AS g
), dimensions AS (
    SELECT *,
        CASE WHEN key_hash %% 10 < 8 THEN key_hash %% 200 ELSE key_hash %% 10000 END AS key_slot,
        %(anchor)s::timestamp - interval '90 days'
            + (time_hash %% 7775900 + (g - 1) %% 8 * 10) * interval '1 second' AS started,
        200 + token_hash %% 30000 AS input_tokens,
        20 + token_hash %% 2000 AS output_tokens,
        100 + token_hash %% 9900 AS duration_ms,
        g %% 100 < 2 AS failed,
        g %% 100 BETWEEN 2 AND 11 AS cached,
        (ARRAY['perf-chat', 'perf-reasoning', 'perf-code'])[1 + g %% 3] AS model_name,
        (ARRAY['openai', 'anthropic', 'bedrock'])[1 + g %% 3] AS provider
    FROM hashes
)
INSERT INTO perf_seed_batch (
    request_id, call_type, api_key, spend, total_tokens, prompt_tokens, completion_tokens,
    "startTime", "endTime", "completionStartTime", model, model_id, model_group,
    custom_llm_provider, "user", team_id, end_user, metadata, request_tags,
    session_id, status, cache_hit, cache_key, litellm_call_id
)
SELECT 'perf-seed-' || g, 'acompletion',
    md5('perf-key:' || key_slot) || md5('perf-key-tail:' || key_slot),
    CASE WHEN failed OR cached THEN 0 ELSE input_tokens * 0.000003 + output_tokens * 0.000015 END,
    CASE WHEN failed THEN 0 ELSE input_tokens + output_tokens END,
    CASE WHEN failed THEN 0 ELSE input_tokens END,
    CASE WHEN failed THEN 0 ELSE output_tokens END,
    started, started + duration_ms * interval '1 millisecond', started + interval '100 milliseconds',
    model_name, 'perf-model-' || g %% 3, model_name, provider,
    'perf-user-' || key_slot / 10, 'perf-team-' || key_slot / 100,
    'perf-end-user-' || key_hash %% 100000,
    jsonb_build_object(
        'user_api_key_alias', 'perf-key-' || key_slot,
        'user_api_key_team_alias', 'perf-team-' || key_slot / 100,
        'user_api_key_user_id', 'perf-user-' || key_slot / 10,
        'user_api_key_team_id', 'perf-team-' || key_slot / 100,
        'user_api_key', md5('perf-key:' || key_slot) || md5('perf-key-tail:' || key_slot),
        'requester_metadata', jsonb_build_object(
            'project', 'project-' || key_slot %% 50,
            'environment', CASE WHEN key_slot %% 5 = 0 THEN 'development' ELSE 'production' END,
            'trace_id', md5('trace:' || g), 'deployment', 'deployment-' || g %% 30),
        'usage_object', jsonb_build_object('prompt_tokens', CASE WHEN failed THEN 0 ELSE input_tokens END,
            'completion_tokens', CASE WHEN failed THEN 0 ELSE output_tokens END),
        'error_information', CASE WHEN failed THEN jsonb_build_object(
            'error_code', '429', 'error_message', 'Synthetic upstream rate limit') ELSE '{}'::jsonb END
    ),
    jsonb_build_array('project-' || key_slot %% 50,
        CASE WHEN key_slot %% 5 = 0 THEN 'development' ELSE 'production' END),
    CASE WHEN g %% 5 <> 0 THEN 'perf-session-' || (g - 1) / 8 ELSE NULL END,
    CASE WHEN failed THEN 'failure' ELSE 'success' END,
    CASE WHEN cached THEN 'True' ELSE 'False' END,
    CASE WHEN cached THEN md5('cache:' || g) ELSE '' END,
    md5('call:' || g)
FROM dimensions
