-- Replaces "LiteLLM_SpendLogs_api_key_startTime_idx" with an index that also
-- carries "spend", so per-key window sums (SUM(spend) WHERE api_key = ... AND
-- "startTime" >= ...) run as index-only scans instead of fetching one heap page
-- per matching row. "spend" is a trailing key column rather than an INCLUDE
-- column because Prisma's @@index cannot express INCLUDE.
--
-- No-op where 20260823000000 already built the covering index. The new index is
-- built before the old one is dropped, so reads keep an index throughout.
--
-- Not CONCURRENTLY on purpose: Postgres rejects CREATE INDEX CONCURRENTLY on a
-- partitioned LiteLLM_SpendLogs (db_scripts/partition_spend_logs.sql) even with
-- IF NOT EXISTS and the index already present, so it would fail this migration
-- on every partitioned install. The blocking build only runs where the earlier
-- (api_key, startTime) version of 20260823000000 was applied. To avoid it,
-- pre-create the index CONCURRENTLY under the same name; the build then no-ops.

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_SpendLogs_api_key_startTime_spend_idx" ON "LiteLLM_SpendLogs"("api_key", "startTime", "spend");

-- DropIndex
DROP INDEX IF EXISTS "LiteLLM_SpendLogs_api_key_startTime_idx";
