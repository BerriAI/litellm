SET LOCAL lock_timeout = '10s';

CREATE TABLE IF NOT EXISTS "LiteLLM_SpendLogReplication" (
    "id" INTEGER PRIMARY KEY CHECK ("id" = 1),
    "region" TEXT NOT NULL,
    "enabled" BOOLEAN NOT NULL DEFAULT true,
    "request_id_hmac_pads" BYTEA NOT NULL CHECK (octet_length("request_id_hmac_pads") = 128)
);

CREATE TABLE IF NOT EXISTS "LiteLLM_SpendLogOutbox" (
    "id" BIGSERIAL PRIMARY KEY,
    "row" JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS "LiteLLM_SpendLogPartitionDrop" (
    "partition_oid" BIGINT PRIMARY KEY,
    "filenode" BIGINT NOT NULL,
    "last_request_id" TEXT,
    "last_start_time" TIMESTAMP(3)
);

CREATE TABLE IF NOT EXISTS "LiteLLM_ReplicatedSpendLogs" (
    "request_id" TEXT NOT NULL,
    "call_type" TEXT NOT NULL,
    "api_key" TEXT NOT NULL DEFAULT '',
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "total_tokens" INTEGER NOT NULL DEFAULT 0,
    "prompt_tokens" INTEGER NOT NULL DEFAULT 0,
    "completion_tokens" INTEGER NOT NULL DEFAULT 0,
    "startTime" TIMESTAMP(3) NOT NULL,
    "endTime" TIMESTAMP(3) NOT NULL,
    "request_duration_ms" INTEGER,
    "completionStartTime" TIMESTAMP(3),
    "model" TEXT NOT NULL DEFAULT '',
    "model_id" TEXT,
    "model_group" TEXT,
    "custom_llm_provider" TEXT,
    "user" TEXT,
    "end_user" TEXT,
    "team_id" TEXT,
    "organization_id" TEXT,
    "status" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL,
    "source_region" TEXT NOT NULL,
    PRIMARY KEY ("request_id", "startTime", "source_region")
);

CREATE INDEX IF NOT EXISTS "LiteLLM_ReplicatedSpendLogs_startTime_idx"
ON "LiteLLM_ReplicatedSpendLogs" ("startTime");
CREATE INDEX IF NOT EXISTS "LiteLLM_ReplicatedSpendLogs_api_key_startTime_idx"
ON "LiteLLM_ReplicatedSpendLogs" ("api_key", "startTime");
CREATE INDEX IF NOT EXISTS "LiteLLM_ReplicatedSpendLogs_end_user_startTime_idx"
ON "LiteLLM_ReplicatedSpendLogs" ("end_user", "startTime");
CREATE INDEX IF NOT EXISTS "LiteLLM_ReplicatedSpendLogs_team_id_startTime_idx"
ON "LiteLLM_ReplicatedSpendLogs" ("team_id", "startTime");
CREATE INDEX IF NOT EXISTS "LiteLLM_ReplicatedSpendLogs_user_startTime_idx"
ON "LiteLLM_ReplicatedSpendLogs" ("user", "startTime");

CREATE TABLE IF NOT EXISTS "LiteLLM_FleetBudgetSpend" (
    "request_id" TEXT NOT NULL,
    "startTime" TIMESTAMP(3) NOT NULL,
    "source_region" TEXT,
    "api_key" TEXT,
    "team_id" TEXT,
    "user" TEXT,
    "end_user" TEXT,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL,
    PRIMARY KEY ("request_id", "startTime")
);

CREATE INDEX IF NOT EXISTS "LiteLLM_FleetBudgetSpend_api_key_startTime_idx"
ON "LiteLLM_FleetBudgetSpend" ("api_key", "startTime") WHERE "source_region" IS NOT NULL;
CREATE INDEX IF NOT EXISTS "LiteLLM_FleetBudgetSpend_team_id_startTime_idx"
ON "LiteLLM_FleetBudgetSpend" ("team_id", "startTime") WHERE "source_region" IS NOT NULL;
CREATE INDEX IF NOT EXISTS "LiteLLM_FleetBudgetSpend_user_startTime_idx"
ON "LiteLLM_FleetBudgetSpend" ("user", "startTime") WHERE "source_region" IS NOT NULL;
CREATE INDEX IF NOT EXISTS "LiteLLM_FleetBudgetSpend_end_user_startTime_idx"
ON "LiteLLM_FleetBudgetSpend" ("end_user", "startTime") WHERE "source_region" IS NOT NULL;

CREATE TABLE IF NOT EXISTS "LiteLLM_FleetKeyBudgetReset" (
    "api_key" TEXT PRIMARY KEY,
    "reset_at" TIMESTAMP(3) NOT NULL
);

CREATE OR REPLACE FUNCTION litellm_replica_request_id(request_id text, call_id text, pads bytea)
RETURNS text LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE SET search_path FROM CURRENT AS $$
DECLARE
    digest bytea;
    extension_schema text;
BEGIN
    IF pads IS NULL OR COALESCE(call_id, '') = ''
        OR (request_id <> call_id AND strpos(request_id, call_id || '_') <> 1) THEN
        RETURN request_id;
    END IF;

    IF current_setting('server_version_num')::integer >= 110000 THEN
        digest := pg_catalog.sha256(
            substring(pads FROM 65) || pg_catalog.sha256(substring(pads FOR 64) || convert_to(request_id, 'UTF8'))
        );
    ELSE
        -- The worker checks for pgcrypto on PostgreSQL 10 before enabling capture.
        SELECT n.nspname INTO STRICT extension_schema
        FROM pg_catalog.pg_extension e
        JOIN pg_catalog.pg_namespace n ON n.oid = e.extnamespace
        WHERE e.extname = 'pgcrypto';
        EXECUTE format(
            'SELECT %1$I.digest(substring($1 FROM 65) || '
            '%1$I.digest(substring($1 FOR 64) || convert_to($2, ''UTF8''), ''sha256''), ''sha256'')',
            extension_schema
        ) INTO digest USING pads, request_id;
    END IF;

    RETURN 'replica:' || encode(digest, 'hex');
END;
$$;

CREATE OR REPLACE FUNCTION litellm_drop_spend_log_partition(partition_name text, batch_size integer)
RETURNS boolean LANGUAGE plpgsql SET search_path FROM CURRENT AS $$
DECLARE
    partition_table regclass := to_regclass(quote_ident(partition_name));
    target_oid bigint := partition_table::oid::bigint;
    partition_filenode bigint := pg_relation_filenode(partition_table)::bigint;
    last_request_id text;
    last_start_time timestamp(3);
    ready boolean;
BEGIN
    IF partition_table IS NULL THEN
        RETURN true;
    END IF;
    SELECT checkpoint.last_request_id IS NULL INTO ready
    FROM "LiteLLM_SpendLogPartitionDrop" AS checkpoint
    WHERE checkpoint.partition_oid = target_oid
        AND checkpoint.filenode = partition_filenode;

    IF ready OR NOT EXISTS (SELECT 1 FROM "LiteLLM_SpendLogReplication" WHERE id = 1) THEN
        -- A queued exclusive parent lock also blocks new writers. Retry on the next cleanup run.
        PERFORM set_config('lock_timeout', '250ms', true);
        LOCK TABLE ONLY "LiteLLM_SpendLogs" IN ACCESS EXCLUSIVE MODE;
        IF NOT EXISTS (
            SELECT 1 FROM pg_inherits
            WHERE inhparent = '"LiteLLM_SpendLogs"'::regclass AND inhrelid = partition_table
        ) THEN
            RETURN true;
        END IF;
        IF EXISTS (SELECT 1 FROM "LiteLLM_SpendLogReplication" WHERE id = 1)
            AND NOT EXISTS (
                SELECT 1 FROM "LiteLLM_SpendLogPartitionDrop" AS checkpoint
                WHERE checkpoint.partition_oid = target_oid
                    AND checkpoint.filenode = pg_relation_filenode(partition_table)::bigint
                    AND checkpoint.last_request_id IS NULL
            ) THEN
            RETURN false;
        END IF;
        EXECUTE format('LOCK TABLE ONLY %s IN ACCESS EXCLUSIVE MODE NOWAIT', partition_table);
        EXECUTE format('DROP TABLE %s', partition_table);
        -- data-migration-ok: runs when retention completes a partition drop
        DELETE FROM "LiteLLM_SpendLogPartitionDrop" AS checkpoint
        WHERE checkpoint.partition_oid = target_oid;
        RETURN true;
    END IF;

    -- Serialize retention and partition DDL while leaving reads and writes through the parent available.
    LOCK TABLE ONLY "LiteLLM_SpendLogs" IN SHARE UPDATE EXCLUSIVE MODE;
    IF NOT EXISTS (
        SELECT 1 FROM pg_inherits
        WHERE inhparent = '"LiteLLM_SpendLogs"'::regclass AND inhrelid = partition_table
    ) THEN
        RETURN true;
    END IF;
    EXECUTE format('LOCK TABLE ONLY %s IN ACCESS SHARE MODE', partition_table);
    partition_filenode := pg_relation_filenode(partition_table)::bigint;
    SELECT checkpoint.last_request_id, checkpoint.last_start_time INTO last_request_id, last_start_time
    FROM "LiteLLM_SpendLogPartitionDrop" AS checkpoint
    WHERE checkpoint.partition_oid = target_oid
        AND checkpoint.filenode = partition_filenode;
    IF FOUND AND last_request_id IS NULL THEN
        RETURN false;
    END IF;

    -- data-migration-ok: runs only on a bounded batch during partition retention
    EXECUTE format($query$
        WITH batch AS (
            SELECT log.request_id, log."startTime", log.created_at, log.updated_at,
                litellm_replica_request_id(log.request_id, log.litellm_call_id, settings.request_id_hmac_pads)
                    AS replica_id
            FROM %s AS log
            JOIN "LiteLLM_SpendLogReplication" AS settings ON settings.id = 1
            WHERE $1 IS NULL OR (log.request_id, log."startTime") > ($1, $2)
            ORDER BY log.request_id, log."startTime"
            LIMIT $3
        ), markers AS (
            INSERT INTO "LiteLLM_FleetBudgetSpend" AS ledger
                ("request_id", "startTime", "created_at", "updated_at")
            SELECT DISTINCT ON (replica_id, "startTime")
                replica_id, "startTime", created_at, updated_at
            FROM batch
            WHERE NOT EXISTS (
                SELECT 1 FROM "LiteLLM_FleetBudgetSpend" AS existing
                WHERE existing.request_id = batch.replica_id AND existing."startTime" = batch."startTime"
                    AND existing.source_region IS NULL
            )
            ORDER BY replica_id, "startTime"
            ON CONFLICT ("request_id", "startTime") DO UPDATE SET "source_region" = NULL,
                "api_key" = NULL, "team_id" = NULL, "user" = NULL, "end_user" = NULL, "spend" = 0
            WHERE ledger.source_region IS NOT NULL
        )
        SELECT request_id, "startTime" FROM batch ORDER BY request_id DESC, "startTime" DESC LIMIT 1
    $query$, partition_table) INTO last_request_id, last_start_time
    USING last_request_id, last_start_time, batch_size;

    -- data-migration-ok: persists this batch's cursor across cleanup runs
    INSERT INTO "LiteLLM_SpendLogPartitionDrop" AS checkpoint
        VALUES (target_oid, partition_filenode, last_request_id, last_start_time)
    ON CONFLICT ON CONSTRAINT "LiteLLM_SpendLogPartitionDrop_pkey" DO UPDATE
        SET "filenode" = EXCLUDED.filenode, "last_request_id" = EXCLUDED.last_request_id,
            "last_start_time" = EXCLUDED.last_start_time;
    RETURN false;
END;
$$;

CREATE OR REPLACE FUNCTION litellm_track_replicated_budget_spend() RETURNS trigger
LANGUAGE plpgsql SET search_path FROM CURRENT AS $$
DECLARE
    local_match boolean;
BEGIN
    IF strpos(NEW.request_id, 'replica:') = 1 THEN
        SELECT EXISTS (
            SELECT 1 FROM "LiteLLM_SpendLogs" AS log
            JOIN "LiteLLM_SpendLogReplication" AS settings ON settings.id = 1
            WHERE log."startTime" = NEW."startTime"
                AND litellm_replica_request_id(log.request_id, log.litellm_call_id, settings.request_id_hmac_pads)
                    = NEW.request_id
        ) INTO local_match;
    ELSE
        SELECT EXISTS (
            SELECT 1 FROM "LiteLLM_SpendLogs" AS log
            WHERE log.request_id = NEW.request_id AND log."startTime" = NEW."startTime"
        ) INTO local_match;
    END IF;
    -- data-migration-ok: runs for the replica that caused the trigger
    INSERT INTO "LiteLLM_FleetBudgetSpend" AS ledger
        ("request_id", "startTime", "source_region", "api_key", "team_id", "user", "end_user",
         "spend", "created_at", "updated_at")
    -- A local match is already covered by the regional spend counters. Keep
    -- its identity in the ledger even after reporting retention deletes it.
    VALUES (NEW.request_id, NEW."startTime",
           CASE WHEN NOT local_match THEN NEW.source_region END,
           CASE WHEN NOT local_match THEN NEW.api_key END,
           CASE WHEN NOT local_match THEN NEW.team_id END,
           CASE WHEN NOT local_match THEN NEW."user" END,
           CASE WHEN NOT local_match THEN NEW.end_user END,
           CASE WHEN NOT local_match THEN NEW.spend ELSE 0 END,
           NEW.created_at, NEW.updated_at)
    ON CONFLICT ("request_id", "startTime") DO UPDATE
    SET "api_key" = EXCLUDED.api_key, "team_id" = EXCLUDED.team_id, "user" = EXCLUDED."user",
        "end_user" = EXCLUDED.end_user, "spend" = EXCLUDED.spend, "updated_at" = EXCLUDED.updated_at
    WHERE ledger.source_region = EXCLUDED.source_region
        AND ledger.created_at = EXCLUDED.created_at AND ledger.updated_at < EXCLUDED.updated_at;
    RETURN NULL;
END;
$$;

CREATE OR REPLACE VIEW "LiteLLM_SpendLogReporting" AS
SELECT
    local_log."request_id", local_log."call_type", local_log."api_key", local_log."spend",
    local_log."total_tokens", local_log."prompt_tokens", local_log."completion_tokens", local_log."startTime",
    local_log."endTime", local_log."request_duration_ms", local_log."completionStartTime", local_log."model",
    local_log."model_id", local_log."model_group", local_log."custom_llm_provider", local_log."user",
    local_log."team_id", local_log."organization_id", local_log."status", local_log."litellm_call_id",
    local_log."created_at", local_log."updated_at", local_log."api_base", local_log."metadata",
    local_log."cache_hit", local_log."cache_key", local_log."request_tags", local_log."end_user",
    local_log."requester_ip_address", local_log."messages", local_log."response", local_log."session_id",
    local_log."mcp_namespaced_tool_name", local_log."billing_agent_id", local_log."agent_id", local_log."proxy_server_request",
    NULL::text AS source_region
FROM "LiteLLM_SpendLogs" AS local_log
UNION ALL
SELECT
    replica."request_id", replica."call_type", replica."api_key", replica."spend",
    replica."total_tokens", replica."prompt_tokens", replica."completion_tokens", replica."startTime",
    replica."endTime", replica."request_duration_ms", replica."completionStartTime", replica."model",
    replica."model_id", replica."model_group", replica."custom_llm_provider", replica."user",
    replica."team_id", replica."organization_id", replica."status", NULL::text,
    replica."created_at", replica."updated_at", NULL::text, NULL::jsonb,
    NULL::text, NULL::text, NULL::jsonb, replica."end_user",
    NULL::text, NULL::jsonb, NULL::jsonb, NULL::text,
    NULL::text, NULL::text, NULL::text, NULL::jsonb,
    replica.source_region
FROM "LiteLLM_ReplicatedSpendLogs" AS replica
WHERE NOT EXISTS (
    SELECT 1 FROM "LiteLLM_SpendLogs" AS local_log WHERE local_log.request_id = replica.request_id
    AND local_log."startTime" = replica."startTime"
)
AND NOT EXISTS (
    SELECT 1 FROM "LiteLLM_SpendLogs" AS log
    JOIN "LiteLLM_SpendLogReplication" AS settings ON settings.id = 1
    WHERE strpos(replica.request_id, 'replica:') = 1 AND log."startTime" = replica."startTime"
        AND litellm_replica_request_id(log.request_id, log.litellm_call_id, settings.request_id_hmac_pads)
            = replica.request_id
);

CREATE OR REPLACE FUNCTION litellm_capture_spend_log() RETURNS trigger
LANGUAGE plpgsql SET search_path FROM CURRENT AS $$
BEGIN
    IF EXISTS (SELECT 1 FROM "LiteLLM_SpendLogReplication" WHERE "id" = 1) THEN
        -- data-migration-ok: tracks only rows changed by the triggering statement
        INSERT INTO "LiteLLM_FleetBudgetSpend" AS ledger
            ("request_id", "startTime", "created_at", "updated_at")
        SELECT DISTINCT ON (1, 2)
            litellm_replica_request_id(log.request_id, log.litellm_call_id, settings.request_id_hmac_pads),
            log."startTime", log.created_at, log.updated_at
        FROM changed_spend_logs AS log
        JOIN "LiteLLM_SpendLogReplication" AS settings ON settings.id = 1
        ON CONFLICT ("request_id", "startTime") DO UPDATE SET "source_region" = NULL,
            "api_key" = NULL, "team_id" = NULL, "user" = NULL, "end_user" = NULL, "spend" = 0
        WHERE ledger.source_region IS NOT NULL;
    END IF;
    IF TG_OP <> 'DELETE' AND current_setting('litellm.spend_log_replica', true) IS DISTINCT FROM 'on'
        AND EXISTS (SELECT 1 FROM "LiteLLM_SpendLogReplication" WHERE "id" = 1 AND "enabled") THEN
        -- data-migration-ok: runs only in the trigger on rows changed by the triggering statement
        INSERT INTO "LiteLLM_SpendLogOutbox" ("row")
        SELECT jsonb_build_object(
            'request_id', litellm_replica_request_id(log.request_id, log.litellm_call_id, settings.request_id_hmac_pads),
            'call_type', log.call_type,
            'api_key', CASE WHEN log.api_key ~ '^[[:xdigit:]]{64}$' THEN log.api_key ELSE '' END,
            'spend', log.spend,
            'total_tokens', log.total_tokens,
            'prompt_tokens', log.prompt_tokens,
            'completion_tokens', log.completion_tokens,
            'startTime', log."startTime",
            'endTime', log."endTime",
            'request_duration_ms', log.request_duration_ms,
            'completionStartTime', log."completionStartTime",
            'model', log.model,
            'model_id', log.model_id,
            'model_group', log.model_group,
            'custom_llm_provider', log.custom_llm_provider,
            'user', log."user",
            'end_user', log.end_user,
            'team_id', log.team_id,
            'organization_id', log.organization_id,
            'status', log.status,
            'created_at', log.created_at,
            'updated_at', log.updated_at
        )
        FROM changed_spend_logs AS log
        JOIN "LiteLLM_SpendLogReplication" AS settings ON settings.id = 1;
    END IF;
    RETURN NULL;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgrelid = '"LiteLLM_ReplicatedSpendLogs"'::regclass AND tgname = 'litellm_track_replicated_budget_spend'
    ) THEN
        CREATE TRIGGER litellm_track_replicated_budget_spend
        AFTER INSERT OR UPDATE ON "LiteLLM_ReplicatedSpendLogs"
        FOR EACH ROW EXECUTE PROCEDURE litellm_track_replicated_budget_spend();
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgrelid = '"LiteLLM_SpendLogs"'::regclass AND tgname = 'litellm_capture_spend_log_insert'
    ) THEN
        CREATE TRIGGER litellm_capture_spend_log_insert
        AFTER INSERT ON "LiteLLM_SpendLogs" REFERENCING NEW TABLE AS changed_spend_logs
        FOR EACH STATEMENT EXECUTE PROCEDURE litellm_capture_spend_log();
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgrelid = '"LiteLLM_SpendLogs"'::regclass AND tgname = 'litellm_capture_spend_log_update'
    ) THEN
        CREATE TRIGGER litellm_capture_spend_log_update
        AFTER UPDATE ON "LiteLLM_SpendLogs" REFERENCING NEW TABLE AS changed_spend_logs
        FOR EACH STATEMENT EXECUTE PROCEDURE litellm_capture_spend_log();
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgrelid = '"LiteLLM_SpendLogs"'::regclass AND tgname = 'litellm_capture_spend_log_delete'
    ) THEN
        CREATE TRIGGER litellm_capture_spend_log_delete
        AFTER DELETE ON "LiteLLM_SpendLogs" REFERENCING OLD TABLE AS changed_spend_logs
        FOR EACH STATEMENT EXECUTE PROCEDURE litellm_capture_spend_log();
    END IF;
END;
$$;
