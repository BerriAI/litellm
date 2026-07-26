-- One-shot backfill of the LiteLLM_DailyToolSpend rollup from the per-request
-- LiteLLM_SpendLogToolIndex x LiteLLM_SpendLogs tables.
--
-- This is an opt-in, manual operation. New deployments do not need it: the
-- rollup is written at request time from the moment the release is deployed.
-- Run it only if you want the Cost Optimization "Spend by tool" card to show
-- history from before the deploy, and only once.
--
-- IMPORTANT caveats before running:
--
-- 1. Pre-deploy index rows may include tools that were merely DECLARED in a
--    request body but never invoked (the release this ships with stops
--    recording those). For agentic clients that declare many tools per
--    request, backfilled history attributes each request's full spend to
--    every declared tool, overstating per-tool spend. Post-deploy rows do not
--    have this problem. If your traffic is mostly such clients, consider not
--    backfilling.
--
-- 2. Coverage is bounded by spend-log retention: rows older than
--    maximum_spend_logs_retention_period are already gone.
--
-- 3. Replace the cutover timestamp below with the time you deployed the
--    release, so backfilled per-request rows cannot double-count on top of
--    rollup rows the new writer already created. ON CONFLICT DO NOTHING is a
--    second guard for (date, tool_name) buckets the writer already touched:
--    such buckets keep the writer's numbers and skip the backfill's.
--
-- Usage:
--   psql "$DATABASE_URL" -v cutover="'2026-07-25T00:00:00Z'" -f db_scripts/backfill_daily_tool_spend.sql

INSERT INTO "LiteLLM_DailyToolSpend" (date, tool_name, spend, total_tokens, request_count, created_at, updated_at)
SELECT
    to_char(ti.start_time, 'YYYY-MM-DD') AS date,
    ti.tool_name,
    COALESCE(SUM(sl.spend), 0) AS spend,
    COALESCE(SUM(sl.total_tokens), 0) AS total_tokens,
    COUNT(*) AS request_count,
    now() AS created_at,
    now() AS updated_at
FROM "LiteLLM_SpendLogToolIndex" ti
JOIN "LiteLLM_SpendLogs" sl ON sl.request_id = ti.request_id
WHERE ti.start_time < :cutover::timestamptz
GROUP BY 1, 2
ON CONFLICT (date, tool_name) DO NOTHING;
