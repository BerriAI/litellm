-- One-shot backfill of LiteLLM_VerificationToken.total_spend (lifetime spend)
-- for keys created before the column was introduced in LiteLLM v1.103.0.
--
-- The column was added with DEFAULT 0 and no backfill, so keys that predate
-- the upgrade report lifetime spend below their current period spend. New
-- deployments do not need this script: total_spend is updated at request
-- time from the moment the release is deployed. Run it only if you want
-- pre-upgrade keys to show their historical lifetime spend, and only once.
-- It is idempotent: every statement uses GREATEST so re-running never lowers
-- a value already written by the proxy.
--
-- IMPORTANT caveats before running:
--
-- 1. Take a backup of the affected tables first:
--      pg_dump "$DATABASE_URL" -t '"LiteLLM_VerificationToken"' -t '"LiteLLM_DeletedVerificationToken"' > key_total_spend_backup.sql
--
-- 2. Statement A (keys without a budget reset) needs no spend logs and is
--    always safe: for these keys the period "spend" column already tracks
--    lifetime usage, so total_spend can never legitimately be below it.
--
-- 3. Statement B (keys with budget_duration, whose "spend" resets each
--    budget period) rebuilds total_spend from LiteLLM_SpendLogs. It requires
--    spend logs to have been enabled, and coverage is bounded by
--    maximum_spend_logs_retention_period: spend older than the retention
--    window is already gone and cannot be recovered. On a large SpendLogs
--    table the GROUP BY scan is slow, so run it off peak.
--
-- 4. No proxy restart is needed. The proxy picks up the corrected values on
--    its next read of each key.
--
-- Usage:
--   psql "$DATABASE_URL" -f db_scripts/backfill_key_total_spend.sql

-- Statement A: keys with no budget reset. "spend" is already lifetime spend.
UPDATE "LiteLLM_VerificationToken"
SET total_spend = GREATEST(total_spend, spend)
WHERE budget_duration IS NULL;

UPDATE "LiteLLM_DeletedVerificationToken"
SET total_spend = GREATEST(total_spend, spend)
WHERE budget_duration IS NULL;

-- Statement B: keys with a budget reset. Rebuild from LiteLLM_SpendLogs,
-- whose api_key column stores the same hashed token as
-- LiteLLM_VerificationToken.token.
UPDATE "LiteLLM_VerificationToken" k
SET total_spend = GREATEST(k.total_spend, s.sum_spend)
FROM (
    SELECT api_key, SUM(spend) AS sum_spend
    FROM "LiteLLM_SpendLogs"
    GROUP BY api_key
) s
WHERE k.token = s.api_key
  AND k.budget_duration IS NOT NULL;

-- Verify: this should return 0.
--   SELECT count(*) FROM "LiteLLM_VerificationToken" WHERE total_spend < spend;
