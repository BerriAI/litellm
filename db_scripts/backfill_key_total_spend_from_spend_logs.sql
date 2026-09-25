-- Optional follow-up to db_scripts/backfill_key_total_spend.sql. Run that
-- script first; this one rebuilds earlier budget periods for the keys it
-- can only partially fix: keys whose spend resets each period, because their own
-- budget_duration IS NOT NULL or because their budget_id links to a
-- LiteLLM_BudgetTable row whose budget_duration IS NOT NULL.
--
-- For those keys the "spend" column only covers the current period, so
-- lifetime spend is reconstructed from LiteLLM_SpendLogs. The join matches
-- l.api_key against both the stored token and its second sha256
-- (encode(sha256(convert_to(token, 'UTF8')), 'hex')), because spend logs
-- written by older paths recorded the re-hashed digest instead of the
-- token. It is idempotent and never lowers a value: every statement only
-- touches rows where total_spend is below the rebuilt sum, so re-running is
-- a no-op, and a key whose log history is shorter than its current period
-- keeps the value backfill_key_total_spend.sql already gave it.
--
-- IMPORTANT caveats before running:
--
-- 1. Take a backup of the affected tables first:
--      pg_dump "$DATABASE_URL" -t '"LiteLLM_VerificationToken"' -t '"LiteLLM_DeletedVerificationToken"' > key_total_spend_backup.sql
--
-- 2. It requires spend logs to have been enabled, and coverage is bounded
--    by maximum_spend_logs_retention_period: spend older than the retention
--    window is already gone and cannot be recovered.
--
-- 3. On a large SpendLogs table the join scan is slow, so run it off peak.
--
-- 4. Run it while the proxy is idle (or with traffic paused). The proxy
--    flushes spend logs in batches, so a request that already raised
--    total_spend but whose log is still queued is missing from the sum, and
--    the rebuilt value would be short by that in-flight amount.
--
-- 5. A custom token can be deleted and recreated, so the archived table can
--    hold several lifetimes of one token. The update only rewrites archived
--    rows that reset, and the log sum covers every lifetime of that token.
--
-- 6. No proxy restart is needed. The proxy picks up the corrected values on
--    its next read of each key.
--
-- Usage:
--   psql "$DATABASE_URL" -f db_scripts/backfill_key_total_spend_from_spend_logs.sql

-- Active keys whose spend resets (own budget_duration, or a linked
-- LiteLLM_BudgetTable row with one). Rebuild from LiteLLM_SpendLogs,
-- matching api_key against the stored token and its second sha256 digest.
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
-- before joining spend logs; the update then hits every resetting archived
-- row.
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
