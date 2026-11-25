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
--   * After partitioning, `prisma db push` (including the proxy's
--     --use_prisma_db_push startup mode) is NOT supported: it tries to rewrite
--     the primary key back to ("request_id"), which Postgres rejects on a
--     partitioned table. The proxy detects this and exits with guidance.
--     Use the default startup path (`prisma migrate deploy`) instead.
--   * Test on a staging copy first and take a backup.
--   * Postgres cannot convert a populated table to partitioned in place, so this
--     renames the old table aside and creates a fresh partitioned table.
--   * Dependent views are repointed to the new table within the transaction.
--     Drop directly dependent materialized views first and recreate afterwards.
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
-- pre-creates upcoming partitions on each run, but its first run can be up to a
-- day away, so this script creates the current and upcoming partitions itself.
-- Set partition_interval and precreate_ahead in the DO block below to your
-- SPEND_LOG_PARTITION_INTERVAL and SPEND_LOG_PARTITION_PRECREATE_AHEAD (default
-- "day" and 7). To roll back, see litellm/scripts/db/unpartition_spend_logs.sql.

BEGIN;

LOCK TABLE "LiteLLM_SpendLogs" IN ACCESS EXCLUSIVE MODE;

-- Capture the definitions before renaming the table so only the relation binding changes.
CREATE TEMP TABLE spend_logs_views_to_repoint ON COMMIT DROP AS
SELECT DISTINCT view_class.oid AS view_oid, view_class.relkind, view_class.reloptions,
    format('%I.%I', view_schema.nspname, view_class.relname) AS qualified_name,
    pg_get_viewdef(view_class.oid) AS definition
FROM pg_depend dep
JOIN pg_rewrite rule ON rule.oid = dep.objid
JOIN pg_class view_class ON view_class.oid = rule.ev_class
JOIN pg_namespace view_schema ON view_schema.oid = view_class.relnamespace
WHERE dep.classid = 'pg_rewrite'::regclass
  AND dep.refclassid = 'pg_class'::regclass
  AND dep.refobjid = '"LiteLLM_SpendLogs"'::regclass
  AND rule.rulename = '_RETURN'
  AND view_class.relkind IN ('v', 'm');

DO $$
DECLARE
    blocked_view regclass;
BEGIN
    IF EXISTS (
        SELECT FROM pg_class spend
        WHERE spend.oid = '"LiteLLM_SpendLogs"'::regclass
          AND (spend.relrowsecurity OR spend.relforcerowsecurity
               OR EXISTS (SELECT FROM pg_policy WHERE polrelid = spend.oid))
    ) THEN
        RAISE EXCEPTION 'LiteLLM_SpendLogs has row-level security settings or policies.'
            USING HINT = 'Stop traffic before removing RLS settings and policies, and restore them on the replacement table before resuming traffic.';
    END IF;

    SELECT view_oid INTO blocked_view
    FROM pg_temp.spend_logs_views_to_repoint
    WHERE relkind = 'm'
    LIMIT 1;
    IF FOUND THEN
        RAISE EXCEPTION 'Materialized view % depends on LiteLLM_SpendLogs. Drop it before running this script and recreate it afterwards.', blocked_view;
    END IF;

    SELECT captured.view_oid INTO blocked_view
    FROM pg_temp.spend_logs_views_to_repoint captured
    JOIN pg_rewrite rule ON rule.ev_class = captured.view_oid AND rule.rulename = '_RETURN'
    JOIN pg_depend dep ON dep.objid = rule.oid AND dep.classid = 'pg_rewrite'::regclass
    JOIN pg_constraint pk ON pk.oid = dep.refobjid AND dep.refclassid = 'pg_constraint'::regclass
    WHERE pk.conrelid = '"LiteLLM_SpendLogs"'::regclass AND pk.contype = 'p'
    LIMIT 1;
    IF FOUND THEN
        RAISE EXCEPTION 'View % depends on the current LiteLLM_SpendLogs primary key.', blocked_view
            USING HINT = 'Rewrite the view to explicitly group or aggregate non-grouped spend log columns before partitioning.';
    END IF;
END $$;

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
ALTER INDEX IF EXISTS "LiteLLM_SpendLogs_api_key_startTime_idx"
    RENAME TO "LiteLLM_SpendLogs_legacy_api_key_startTime_idx";
ALTER INDEX IF EXISTS "LiteLLM_SpendLogs_litellm_call_id_idx"
    RENAME TO "LiteLLM_SpendLogs_legacy_litellm_call_id_idx";

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

CREATE INDEX IF NOT EXISTS "LiteLLM_SpendLogs_api_key_startTime_idx"
    ON "LiteLLM_SpendLogs" ("api_key", "startTime");

-- Safety net: any row whose startTime has no explicit partition lands here so
-- writes never fail. The cleanup job never drops the DEFAULT partition.
CREATE TABLE IF NOT EXISTS "LiteLLM_SpendLogs_pdefault"
    PARTITION OF "LiteLLM_SpendLogs" DEFAULT;

-- Create the current period and the next precreate_ahead periods with the same
-- names and bounds the cleanup job uses, so no write lands in DEFAULT. A row in
-- DEFAULT blocks creating the partition that covers it, and every later
-- partition creation scans DEFAULT while holding an ACCESS EXCLUSIVE lock on
-- "LiteLLM_SpendLogs". To backfill, set oldest_day to the oldest "startTime"
-- you will copy so those days get partitions too.
DO $$
DECLARE
    partition_interval CONSTANT text := 'day';
    precreate_ahead CONSTANT int := 7;
    today CONSTANT date := (now() AT TIME ZONE 'UTC')::date;
    oldest_day CONSTANT date := today;
    step interval;
    lower_bound date;
    last_lower_bound date;
BEGIN
    IF partition_interval NOT IN ('day', 'week', 'month') THEN
        RAISE EXCEPTION 'partition_interval must be day, week or month, got %', partition_interval;
    END IF;
    step := ('1 ' || partition_interval)::interval;
    lower_bound := date_trunc(partition_interval, oldest_day::timestamp)::date;
    last_lower_bound := (date_trunc(partition_interval, today::timestamp) + precreate_ahead * step)::date;
    WHILE lower_bound <= last_lower_bound LOOP
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I PARTITION OF "LiteLLM_SpendLogs" FOR VALUES FROM (%L) TO (%L)',
            'LiteLLM_SpendLogs_p' || to_char(lower_bound, 'YYYYMMDD'),
            lower_bound,
            (lower_bound + step)::date
        );
        lower_bound := (lower_bound + step)::date;
    END LOOP;
END $$;

-- Preserve view OIDs so views of these views keep working without recreation.
DO $$
DECLARE
    view_name text;
    view_options text[];
    view_definition text;
    previous_lock_timeout text := current_setting('lock_timeout');
BEGIN
    -- Readers can hold a view lock while waiting for the spend table lock.
    PERFORM set_config('lock_timeout', '1ms', true);
    FOR view_name, view_options, view_definition IN
        SELECT qualified_name, reloptions, definition
        FROM pg_temp.spend_logs_views_to_repoint
    LOOP
        EXECUTE format('CREATE OR REPLACE VIEW %s %s AS %s', view_name,
            CASE WHEN view_options IS NULL THEN ''
                ELSE format('WITH (%s)', array_to_string(view_options, ', '))
            END,
            view_definition);
    END LOOP;
    PERFORM set_config('lock_timeout', previous_lock_timeout, true);
EXCEPTION WHEN lock_not_available OR deadlock_detected THEN
    RAISE EXCEPTION 'Could not rebind spend log view % while a conflicting lock is held.', view_name
        USING ERRCODE = '55P03', HINT = 'Retry when queries against spend log views have stopped.';
END $$;

COMMIT;

-- Backfill (optional). Rows route to the correct partition automatically.
-- For large legacy tables, copy in time-bounded batches during a low-traffic
-- window instead of one statement, or simply keep "LiteLLM_SpendLogs_legacy"
-- read-only until its data ages past your retention, then DROP it.
-- Views now read the new table, so legacy rows appear only after backfilling.
--
-- Backfilled rows older than oldest_day land in the DEFAULT partition, where
-- they stay until retention deletes them row by row, and every partition the
-- cleanup job creates has to scan them first. Set oldest_day above before
-- running the swap so every backfilled day has its own partition.
--
--   INSERT INTO "LiteLLM_SpendLogs"
--   SELECT * FROM "LiteLLM_SpendLogs_legacy"
--   WHERE "startTime" >= now() - interval '30 days';
--
--   DROP TABLE "LiteLLM_SpendLogs_legacy";
