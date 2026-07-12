-- Converts an existing LiteLLM_SpendLogs table into a native Postgres
-- range-partitioned table keyed on "startTime".
--
-- Why: at high request volume, retention via DELETE leaves dead tuples that
-- autovacuum cannot reclaim quickly enough, so the table keeps growing on disk
-- (seen at 450GB+ after ~1 month). With partitioning, retention drops whole
-- partitions, which is instant and returns disk to the OS immediately.
--
-- This is an opt-in, manual operation. The default LiteLLM schema is NOT
-- partitioned, so existing installs are unaffected until you run this.
--
-- IMPORTANT
--   * Test on a staging copy first and take a backup.
--   * Postgres cannot convert a populated table to partitioned in place, so this
--     renames the old table aside and creates a fresh partitioned table.
--   * The partition key ("startTime") must be part of the primary key, so the
--     PK becomes composite ("request_id", "startTime"). LiteLLM's write path uses
--     INSERT ... ON CONFLICT DO NOTHING, which is compatible with this.
--   * Choose a partition granularity ("day" is the recommended default for
--     high-volume tables) and keep it consistent with SPEND_LOG_PARTITION_INTERVAL.
--
-- After running this, enable the feature and set a retention period in
-- proxy_config.yaml:
--   general_settings:
--     use_spend_logs_partitioning: true
--     maximum_spend_logs_retention_period: "30d"
-- The spend-log cleanup job then verifies the table is partitioned and reclaims
-- disk by dropping expired partitions instead of deleting rows. It also
-- pre-creates upcoming partitions on each run. To roll back, see
-- db_scripts/unpartition_spend_logs.sql.

BEGIN;

ALTER TABLE "LiteLLM_SpendLogs" RENAME TO "LiteLLM_SpendLogs_legacy";

-- Renaming a table does NOT rename its indexes, and index names are unique per
-- schema. Move the legacy table's indexes aside so the CREATE INDEX statements
-- below actually create indexes on the new partitioned table instead of being
-- silently skipped by IF NOT EXISTS, and so the new PK keeps the canonical
-- name instead of getting a "_pkey1" suffix.
ALTER INDEX IF EXISTS "LiteLLM_SpendLogs_pkey"
    RENAME TO "LiteLLM_SpendLogs_legacy_pkey";
ALTER INDEX IF EXISTS "LiteLLM_SpendLogs_startTime_idx"
    RENAME TO "LiteLLM_SpendLogs_legacy_startTime_idx";
ALTER INDEX IF EXISTS "LiteLLM_SpendLogs_startTime_request_id_idx"
    RENAME TO "LiteLLM_SpendLogs_legacy_startTime_request_id_idx";
ALTER INDEX IF EXISTS "LiteLLM_SpendLogs_end_user_idx"
    RENAME TO "LiteLLM_SpendLogs_legacy_end_user_idx";
ALTER INDEX IF EXISTS "LiteLLM_SpendLogs_session_id_idx"
    RENAME TO "LiteLLM_SpendLogs_legacy_session_id_idx";

CREATE TABLE "LiteLLM_SpendLogs" (
    LIKE "LiteLLM_SpendLogs_legacy" INCLUDING DEFAULTS INCLUDING GENERATED
) PARTITION BY RANGE ("startTime");

ALTER TABLE "LiteLLM_SpendLogs"
    ADD PRIMARY KEY ("request_id", "startTime");

-- Recreate every index Prisma defines on the table. LIKE ... INCLUDING DEFAULTS
-- INCLUDING GENERATED copies columns and defaults but NOT indexes, so without
-- these the admin-UI cost-reporting queries that filter by end_user/session_id
-- fall back to sequential scans. On a partitioned parent these propagate to
-- every current and future partition automatically.
CREATE INDEX IF NOT EXISTS "LiteLLM_SpendLogs_startTime_idx"
    ON "LiteLLM_SpendLogs" ("startTime");

CREATE INDEX IF NOT EXISTS "LiteLLM_SpendLogs_startTime_request_id_idx"
    ON "LiteLLM_SpendLogs" ("startTime", "request_id");

CREATE INDEX IF NOT EXISTS "LiteLLM_SpendLogs_end_user_idx"
    ON "LiteLLM_SpendLogs" ("end_user");

CREATE INDEX IF NOT EXISTS "LiteLLM_SpendLogs_session_id_idx"
    ON "LiteLLM_SpendLogs" ("session_id");

-- Safety net: any row whose startTime has no explicit partition lands here so
-- writes never fail. The cleanup job never drops the DEFAULT partition.
CREATE TABLE IF NOT EXISTS "LiteLLM_SpendLogs_pdefault"
    PARTITION OF "LiteLLM_SpendLogs" DEFAULT;

COMMIT;

-- Backfill (optional). Rows route to the correct partition automatically.
-- For large legacy tables, copy in time-bounded batches during a low-traffic
-- window instead of one statement, or simply keep "LiteLLM_SpendLogs_legacy"
-- read-only until its data ages past your retention, then DROP it.
--
-- Backfilled rows land in the DEFAULT partition until explicit partitions
-- cover their dates. Postgres refuses to create a partition whose range
-- overlaps rows already in DEFAULT, so the cleanup job may log a warning when
-- pre-creating today's partition right after a backfill; it recovers on its
-- own once those dates age out, and future partitions are unaffected because
-- they are always created ahead of writes.
--
--   INSERT INTO "LiteLLM_SpendLogs"
--   SELECT * FROM "LiteLLM_SpendLogs_legacy"
--   WHERE "startTime" >= now() - interval '30 days';
--
--   DROP TABLE "LiteLLM_SpendLogs_legacy";
