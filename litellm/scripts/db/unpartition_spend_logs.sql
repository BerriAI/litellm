-- Rolls back litellm/scripts/db/partition_spend_logs.sql: converts the native
-- range-partitioned "LiteLLM_SpendLogs" table back into a plain,
-- non-partitioned table matching the default LiteLLM schema.
--
-- When/why: run this if you want to stop using partition-based retention and
-- return to DELETE-based cleanup, or to restore the original single-column
-- primary key ("request_id") that the partitioned layout had to widen to a
-- composite ("request_id", "startTime").
--
-- IMPORTANT
--   * Test on a staging copy first and take a backup.
--   * Postgres cannot convert a partitioned table back in place, so this
--     renames the partitioned table aside and creates a fresh plain table.
--   * Dependent views are repointed before dropping the partitioned table.
--     Drop directly dependent materialized views first and recreate afterwards.
--   * The composite PK could in principle hold the same "request_id" in more
--     than one partition, so rows are copied with ON CONFLICT DO NOTHING to
--     restore the single-column PK without failing on such duplicates.
--   * For large tables the INSERT ... SELECT copies every surviving row and may
--     run long; do it during a low-traffic window.
--   * Also remove use_spend_logs_partitioning from proxy_config.yaml (or set it
--     to false) so the cleanup job returns to DELETE-based retention.

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
END $$;

ALTER TABLE "LiteLLM_SpendLogs" RENAME TO "LiteLLM_SpendLogs_partitioned";

-- Renaming a table does NOT rename its indexes, and index names are unique per
-- schema. Move the partitioned table's indexes aside so the CREATE INDEX
-- statements below actually create indexes on the new plain table instead of
-- being silently skipped by IF NOT EXISTS, and so the new PK keeps the
-- canonical name.
ALTER INDEX IF EXISTS "LiteLLM_SpendLogs_pkey"
    RENAME TO "LiteLLM_SpendLogs_partitioned_pkey";
ALTER INDEX IF EXISTS "LiteLLM_SpendLogs_pkey1"
    RENAME TO "LiteLLM_SpendLogs_partitioned_pkey1";
ALTER INDEX IF EXISTS "LiteLLM_SpendLogs_startTime_idx"
    RENAME TO "LiteLLM_SpendLogs_partitioned_startTime_idx";
ALTER INDEX IF EXISTS "LiteLLM_SpendLogs_startTime_request_id_idx"
    RENAME TO "LiteLLM_SpendLogs_partitioned_startTime_request_id_idx";
ALTER INDEX IF EXISTS "LiteLLM_SpendLogs_end_user_idx"
    RENAME TO "LiteLLM_SpendLogs_partitioned_end_user_idx";
ALTER INDEX IF EXISTS "LiteLLM_SpendLogs_session_id_idx"
    RENAME TO "LiteLLM_SpendLogs_partitioned_session_id_idx";
ALTER INDEX IF EXISTS "LiteLLM_SpendLogs_api_key_startTime_idx"
    RENAME TO "LiteLLM_SpendLogs_partitioned_api_key_startTime_idx";

CREATE TABLE "LiteLLM_SpendLogs" (
    LIKE "LiteLLM_SpendLogs_partitioned" INCLUDING DEFAULTS INCLUDING GENERATED
);

ALTER TABLE "LiteLLM_SpendLogs"
    ADD PRIMARY KEY ("request_id");

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

INSERT INTO "LiteLLM_SpendLogs"
SELECT * FROM "LiteLLM_SpendLogs_partitioned"
ON CONFLICT ("request_id") DO NOTHING;

DROP TABLE "LiteLLM_SpendLogs_partitioned";

COMMIT;
