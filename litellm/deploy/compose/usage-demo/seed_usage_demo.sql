-- Synthetic Usage dashboard history for the disposable litellm-usage-demo database only.
-- It refuses to run anywhere else, and rerunning replaces the usage-demo namespace rows.
-- These are aggregate rows written directly, so they exercise the read path and UI, not ingestion.
--
-- Twelve calendar months of daily aggregates ending today (UTC), so the current
-- month is partial. Deterministic: rerunning on the same day writes identical rows.
--   * twelve model groups whose mix shifts over the year, so the top 8 + Other matters
--   * cached prompt tokens as a subset of prompt tokens, never added on top
--   * demo-image-gen: per-request spend with zero tokens
--   * demo-subscription-chat: tokens with zero metered spend
--   * no traffic at all in the fifth month, a real zero bucket
--   * demo-legacy-completion stops after month eight, demo-coding-assistant starts in month four
--   * the coding key's traffic is also tagged as Claude Code, so the client panel has data

DO $$
BEGIN
  IF current_setting('cluster_name') IS DISTINCT FROM 'litellm_usage_demo' THEN
    RAISE EXCEPTION 'refusing to run outside the litellm-usage-demo database (cluster_name is "%")',
      current_setting('cluster_name');
  END IF;
END
$$;

BEGIN;

DELETE FROM "LiteLLM_DailyUserSpend"
WHERE api_key LIKE 'usage-demo-key-%' AND user_id LIKE 'usage-demo-user-%';

INSERT INTO "LiteLLM_UserTable" (user_id, user_alias, user_email, user_role, models)
VALUES
  ('usage-demo-user-a', 'Synthetic demo: support and batch', 'usage-demo-a@example.invalid', 'internal_user', '{}'),
  ('usage-demo-user-b', 'Synthetic demo: research and coding', 'usage-demo-b@example.invalid', 'internal_user', '{}')
ON CONFLICT (user_id) DO NOTHING;

WITH bounds AS (
  SELECT
    (now() AT TIME ZONE 'UTC')::date AS today,
    (date_trunc('month', (now() AT TIME ZONE 'UTC')::date) - interval '11 months')::date AS first_day
),
days AS (
  SELECT
    d::date AS day,
    (extract(year FROM d) * 12 + extract(month FROM d)
      - extract(year FROM b.first_day) * 12 - extract(month FROM b.first_day))::int AS month_index,
    (d::date - b.first_day) AS day_index,
    extract(isodow FROM d)::int AS weekday
  FROM bounds b, generate_series(b.first_day, b.today, interval '1 day') AS d
),
models (
  seed, model_group, model, provider, api_key, user_id, requests, prompt_per_request, completion_per_request,
  cache_share, usd_per_m_in, usd_per_m_out, usd_per_request, ramp, first_month, last_month
) AS (
  VALUES
    (1, 'demo-chat-large', 'openai/demo-chat-large', 'openai', 'usage-demo-key-support', 'usage-demo-user-a',
      900, 1800, 450, 0.35, 2.50, 10.00, 0.0, 0.2, 0, 11),
    (2, 'demo-chat-fast', 'openai/demo-chat-fast', 'openai', 'usage-demo-key-support', 'usage-demo-user-a',
      2600, 900, 220, 0.20, 0.15, 0.60, 0.0, 0.6, 0, 11),
    (3, 'demo-reasoning', 'anthropic/demo-reasoning', 'anthropic', 'usage-demo-key-research', 'usage-demo-user-b',
      220, 6000, 2400, 0.55, 3.00, 15.00, 0.0, 1.4, 0, 11),
    (4, 'demo-coding-assistant', 'anthropic/demo-coding-assistant', 'anthropic', 'usage-demo-key-coding',
      'usage-demo-user-b', 600, 14000, 900, 0.70, 3.00, 15.00, 0.0, 3.0, 3, 11),
    (5, 'demo-embeddings', 'openai/demo-embeddings', 'openai', 'usage-demo-key-batch', 'usage-demo-user-a',
      4000, 600, 0, 0.0, 0.02, 0.0, 0.0, 0.3, 0, 11),
    (6, 'demo-image-gen', 'openai/demo-image-gen', 'openai', 'usage-demo-key-support', 'usage-demo-user-a',
      60, 0, 0, 0.0, 0.0, 0.0, 0.04, 0.5, 0, 11),
    (7, 'demo-subscription-chat', 'chatgpt/demo-subscription-chat', 'chatgpt', 'usage-demo-key-coding',
      'usage-demo-user-b', 300, 3000, 700, 0.30, 0.0, 0.0, 0.0, 1.0, 0, 11),
    (8, 'demo-batch-summarizer', 'bedrock/demo-batch-summarizer', 'bedrock', 'usage-demo-key-batch',
      'usage-demo-user-a', 150, 9000, 600, 0.0, 0.80, 4.00, 0.0, 0.0, 0, 11),
    (9, 'demo-vision', 'gemini/demo-vision', 'gemini', 'usage-demo-key-research', 'usage-demo-user-b',
      120, 2500, 300, 0.0, 1.25, 5.00, 0.0, 0.4, 0, 11),
    (10, 'demo-legacy-completion', 'azure/demo-legacy-completion', 'azure', 'usage-demo-key-support',
      'usage-demo-user-a', 700, 1200, 300, 0.0, 1.50, 2.00, 0.0, -0.9, 0, 7),
    (11, 'demo-eu-regional-assistant-with-a-deliberately-long-model-group-name',
      'vertex_ai/demo-eu-regional-assistant-with-a-deliberately-long-model-group-name', 'vertex_ai',
      'usage-demo-key-research', 'usage-demo-user-b', 40, 2000, 400, 0.20, 1.25, 5.00, 0.0, 0.2, 0, 11),
    (12, 'demo-moderation', 'openai/demo-moderation', 'openai', 'usage-demo-key-support', 'usage-demo-user-a',
      500, 300, 0, 0.0, 0.01, 0.0, 0.0, 0.0, 0, 11)
),
volumes AS (
  SELECT
    m.*,
    d.day,
    greatest(
      1,
      round(
        m.requests
        * greatest(0.05, 1 + m.ramp * d.month_index / 11.0)
        * (CASE WHEN d.weekday IN (6, 7) THEN 0.45 ELSE 1.0 END)
        * (1 + 0.15 * sin(d.day_index * 1.7 + m.seed))
      )
    )::bigint AS api_requests
  FROM models m
  JOIN days d ON d.month_index BETWEEN m.first_month AND m.last_month
  WHERE d.month_index <> 4
),
daily_usage AS (
  SELECT
    v.*,
    (v.api_requests * v.prompt_per_request)::bigint AS prompt_tokens,
    (v.api_requests * v.completion_per_request)::bigint AS completion_tokens,
    round(v.api_requests * v.prompt_per_request * v.cache_share)::bigint AS cache_read_input_tokens,
    round(v.api_requests * v.prompt_per_request * v.cache_share * 0.1)::bigint AS cache_creation_input_tokens,
    (v.api_requests * 2 / 100)::bigint AS failed_requests
  FROM volumes v
)
INSERT INTO "LiteLLM_DailyUserSpend" (
  id, user_id, date, api_key, model, model_group, custom_llm_provider,
  prompt_tokens, completion_tokens, cache_read_input_tokens, cache_creation_input_tokens,
  spend, api_requests, successful_requests, failed_requests, total_response_time_ms, timed_requests,
  created_at, updated_at
)
SELECT
  'usage-demo-' || md5(u.user_id || u.api_key || u.model || u.day::text),
  u.user_id,
  to_char(u.day, 'YYYY-MM-DD'),
  u.api_key,
  u.model,
  u.model_group,
  u.provider,
  u.prompt_tokens,
  u.completion_tokens,
  u.cache_read_input_tokens,
  u.cache_creation_input_tokens,
  round((
    (u.prompt_tokens - u.cache_read_input_tokens) * u.usd_per_m_in / 1e6
    + u.cache_read_input_tokens * u.usd_per_m_in * 0.1 / 1e6
    + u.completion_tokens * u.usd_per_m_out / 1e6
    + u.api_requests * u.usd_per_request
  )::numeric, 6)::double precision,
  u.api_requests,
  u.api_requests - u.failed_requests,
  u.failed_requests,
  u.api_requests * (400 + 30 * u.seed),
  u.api_requests,
  now(),
  now()
FROM daily_usage u;

DELETE FROM "LiteLLM_DailyTagSpend" WHERE api_key LIKE 'usage-demo-key-%';

INSERT INTO "LiteLLM_DailyTagSpend" (
  id, tag, date, api_key, model, model_group, custom_llm_provider,
  prompt_tokens, completion_tokens, cache_read_input_tokens, cache_creation_input_tokens,
  spend, api_requests, successful_requests, failed_requests, total_response_time_ms, timed_requests,
  created_at, updated_at
)
SELECT
  'usage-demo-tag-' || md5(id),
  'User-Agent: claude-cli/2.1.263 (external, cli)',
  date, api_key, model, model_group, custom_llm_provider,
  prompt_tokens, completion_tokens, cache_read_input_tokens, cache_creation_input_tokens,
  spend, api_requests, successful_requests, failed_requests, total_response_time_ms, timed_requests,
  now(), now()
FROM "LiteLLM_DailyUserSpend"
WHERE api_key = 'usage-demo-key-coding' AND user_id = 'usage-demo-user-b';

-- Gateway request tallies for the same synthetic days, so the request tiles that prefer
-- them agree with the spend aggregate. The route name keeps these rows in the demo namespace.
DELETE FROM "LiteLLM_DailyGatewayRequests" WHERE route LIKE 'usage-demo:%';

INSERT INTO "LiteLLM_DailyGatewayRequests" (date, category, route, successful_requests, failed_requests, created_at, updated_at)
SELECT date, 'llm', 'usage-demo:/chat/completions', sum(successful_requests), sum(failed_requests), now(), now()
FROM "LiteLLM_DailyUserSpend"
WHERE api_key LIKE 'usage-demo-key-%' AND user_id LIKE 'usage-demo-user-%'
GROUP BY date;

COMMIT;

SELECT
  count(*) AS demo_rows,
  min(date) AS first_day,
  max(date) AS last_day,
  round(sum(spend)::numeric, 2) AS demo_spend_usd,
  sum(prompt_tokens + completion_tokens) AS demo_total_tokens
FROM "LiteLLM_DailyUserSpend"
WHERE api_key LIKE 'usage-demo-key-%';
