-- One-shot backfill of LiteLLM_VerificationToken.total_spend (lifetime spend)
-- for keys created before the column was introduced in LiteLLM v1.103.0.
--
-- The column was added with DEFAULT 0 and no backfill, so keys that predate
-- the upgrade report lifetime spend below their current period spend. New
-- deployments do not need this script: total_spend is updated at request
-- time from the moment the release is deployed. Run it only if you want
-- pre-upgrade keys to show their historical lifetime spend. It is
-- idempotent: every statement only touches rows where total_spend is below
-- the value it would be set to, so re-running is a no-op. It touches no
-- spend logs and runs in seconds.
--
-- IMPORTANT caveats before running:
--
-- 1. Take a backup of the affected tables first:
--      pg_dump "$DATABASE_URL" -t '"LiteLLM_VerificationToken"' -t '"LiteLLM_DeletedVerificationToken"' > key_total_spend_backup.sql
--
-- 2. A key "resets" when its own budget_duration IS NOT NULL, or when its
--    budget_id links to a LiteLLM_BudgetTable row whose budget_duration IS
--    NOT NULL (a linked budget resets the key's spend each period too). This
--    script only fixes non-resetting keys, whose "spend" column already
--    tracks lifetime usage. Resetting keys are left alone here because their
--    "spend" is only the current period; for them,
--    db_scripts/backfill_key_total_spend_from_spend_logs.sql is an optional
--    follow-up that rebuilds their history from LiteLLM_SpendLogs.
--
-- 3. No proxy restart is needed. The proxy picks up the corrected values on
--    its next read of each key.
--
-- Usage:
--   psql "$DATABASE_URL" -f db_scripts/backfill_key_total_spend.sql

-- Keys with no budget reset (own or via a linked budget): "spend" is
-- already lifetime spend.
UPDATE "LiteLLM_VerificationToken"
SET total_spend = spend
WHERE total_spend < spend
  AND budget_duration IS NULL
  AND (budget_id IS NULL OR budget_id NOT IN (
      SELECT budget_id FROM "LiteLLM_BudgetTable" WHERE budget_duration IS NOT NULL
  ));

UPDATE "LiteLLM_DeletedVerificationToken"
SET total_spend = spend
WHERE total_spend < spend
  AND budget_duration IS NULL
  AND (budget_id IS NULL OR budget_id NOT IN (
      SELECT budget_id FROM "LiteLLM_BudgetTable" WHERE budget_duration IS NOT NULL
  ));

-- Verify: non-resetting keys should return 0. Resetting keys are excluded
-- because their current period spend is not comparable to lifetime spend.
--   SELECT count(*) FROM "LiteLLM_VerificationToken"
--   WHERE total_spend < spend AND budget_duration IS NULL
--     AND (budget_id IS NULL OR budget_id NOT IN (
--         SELECT budget_id FROM "LiteLLM_BudgetTable" WHERE budget_duration IS NOT NULL));
