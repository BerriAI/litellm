-- One-shot backfill of LiteLLM_VerificationToken.total_spend (lifetime spend)
-- for keys created before the column was introduced in LiteLLM v1.103.0.
--
-- The column was added with DEFAULT 0 and no backfill, so keys that predate
-- the upgrade report lifetime spend below their current period spend. New
-- deployments do not need this script: total_spend is updated at request
-- time from the moment the release is deployed. Run it only if you want
-- pre-upgrade keys to show their historical lifetime spend. It sets lifetime
-- spend to at least the current spend on every key, active and archived,
-- because current period spend is a valid lower bound on lifetime spend.
-- For keys with no budget reset that is already the exact lifetime value;
-- for resetting keys it only recovers the current period. It is idempotent:
-- it only touches rows where total_spend is below spend, so re-running is a
-- no-op. It touches no spend logs and runs in seconds.
--
-- IMPORTANT caveats before running:
--
-- 1. Take a backup of the affected tables first:
--      pg_dump "$DATABASE_URL" -t '"LiteLLM_VerificationToken"' -t '"LiteLLM_DeletedVerificationToken"' > key_total_spend_backup.sql
--
-- 2. A key "resets" when its own budget_duration IS NOT NULL, or when its
--    budget_id links to a LiteLLM_BudgetTable row whose budget_duration IS
--    NOT NULL (a linked budget resets the key's spend each period too). For
--    those keys this script only recovers the current period;
--    db_scripts/backfill_key_total_spend_from_spend_logs.sql is an optional
--    follow-up that rebuilds the earlier periods from LiteLLM_SpendLogs.
--
-- 3. No proxy restart is needed. The proxy picks up the corrected values on
--    its next read of each key.
--
-- Usage:
--   psql "$DATABASE_URL" -f db_scripts/backfill_key_total_spend.sql

UPDATE "LiteLLM_VerificationToken"
SET total_spend = spend
WHERE total_spend < spend;

UPDATE "LiteLLM_DeletedVerificationToken"
SET total_spend = spend
WHERE total_spend < spend;

-- Verify: this should return 0.
--   SELECT count(*) FROM "LiteLLM_VerificationToken" WHERE total_spend < spend;
