-- One-shot backfill of LiteLLM_VerificationToken.total_spend (lifetime spend)
-- for keys created before the column was introduced in LiteLLM v1.103.0.
--
-- The column was added with DEFAULT 0 and no backfill, so keys that predate
-- the upgrade report lifetime spend below their current period spend. New
-- deployments do not need this script: total_spend is updated at request
-- time from the moment the release is deployed. Run it only if you want
-- pre-upgrade keys to show their historical lifetime spend. It is
-- idempotent: every statement only touches rows where total_spend is below
-- the value it would be set to, so re-running is a no-op.
--
-- IMPORTANT caveats before running:
--
-- 1. Take a backup of the affected tables first:
--      pg_dump "$DATABASE_URL" -t '"LiteLLM_VerificationToken"' -t '"LiteLLM_DeletedVerificationToken"' > key_total_spend_backup.sql
--
-- 2. A key "resets" when its own budget_duration IS NOT NULL, or when its
--    budget_id links to a LiteLLM_BudgetTable row whose budget_duration IS
--    NOT NULL (a linked budget resets the key's spend each period too).
--    Statement A covers non-resetting keys: their "spend" column already
--    tracks lifetime usage, so total_spend can be lifted straight from it.
--
-- 3. Statement B covers resetting keys, whose "spend" restarts each budget
--    period. It rebuilds total_spend from LiteLLM_SpendLogs. The join
--    matches l.api_key against both the stored token and its second sha256
--    (encode(sha256(convert_to(token, 'UTF8')), 'hex')), because spend logs
--    written by older paths recorded the re-hashed digest instead of the
--    token. It requires spend logs to have been enabled, and coverage is
--    bounded by maximum_spend_logs_retention_period: spend older than the
--    retention window is already gone and cannot be recovered. On a large
--    SpendLogs table the join scan is slow, so run it off peak.
--
--    Run Statement B while the proxy is idle (or with traffic paused). The
--    proxy flushes spend logs in batches, so a request that already raised
--    total_spend but whose log is still queued is missing from the sum, and
--    the rebuilt value would be short by that in-flight amount.
--
--    A custom token can be deleted and recreated, so the archived table can
--    hold several lifetimes of one token. Statement B only rewrites archived
--    rows that reset, and the log sum covers every lifetime of that token.
--
-- 4. No proxy restart is needed. The proxy picks up the corrected values on
--    its next read of each key.
--
-- Usage:
--   psql "$DATABASE_URL" -f db_scripts/backfill_key_total_spend.sql

-- Statement A: keys with no budget reset (own or via a linked budget).
-- "spend" is already lifetime spend.
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

-- Statement B: keys whose spend resets (own budget_duration, or a linked
-- LiteLLM_BudgetTable row with one). Rebuild from LiteLLM_SpendLogs, matching
-- api_key against the stored token and its second sha256 digest.
UPDATE "LiteLLM_VerificationToken" k
SET total_spend = s.sum_spend
FROM (
    SELECT k2.token, SUM(l.spend) AS sum_spend
    FROM "LiteLLM_VerificationToken" k2
    JOIN "LiteLLM_SpendLogs" l
      ON l.api_key IN (k2.token, encode(sha256(convert_to(k2.token, 'UTF8')), 'hex'))
    WHERE k2.budget_duration IS NOT NULL
       OR k2.budget_id IN (
           SELECT budget_id FROM "LiteLLM_BudgetTable" WHERE budget_duration IS NOT NULL
       )
    GROUP BY k2.token
) s
WHERE k.token = s.token
  AND k.total_spend < s.sum_spend;

-- Archived tokens are not unique, so collapse them to one row per token
-- before joining spend logs; the update then hits every resetting archived row.
UPDATE "LiteLLM_DeletedVerificationToken" k
SET total_spend = s.sum_spend
FROM (
    SELECT k2.token, SUM(l.spend) AS sum_spend
    FROM (
        SELECT DISTINCT token
        FROM "LiteLLM_DeletedVerificationToken"
        WHERE budget_duration IS NOT NULL
           OR budget_id IN (
               SELECT budget_id FROM "LiteLLM_BudgetTable" WHERE budget_duration IS NOT NULL
           )
    ) k2
    JOIN "LiteLLM_SpendLogs" l
      ON l.api_key IN (k2.token, encode(sha256(convert_to(k2.token, 'UTF8')), 'hex'))
    GROUP BY k2.token
) s
WHERE k.token = s.token
  AND k.total_spend < s.sum_spend
  AND (k.budget_duration IS NOT NULL
       OR k.budget_id IN (
           SELECT budget_id FROM "LiteLLM_BudgetTable" WHERE budget_duration IS NOT NULL
       ));

-- Verify: this should return 0.
--   SELECT count(*) FROM "LiteLLM_VerificationToken" WHERE total_spend < spend;
