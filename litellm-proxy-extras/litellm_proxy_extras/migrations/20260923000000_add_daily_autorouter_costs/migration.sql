BEGIN;
SET LOCAL lock_timeout = '10s';
SET LOCAL statement_timeout = '30s';

DO $install$
DECLARE
    target_schema CONSTANT TEXT := current_schema();
    daily_table CONSTANT REGCLASS := to_regclass(format('%I.%I', target_schema, 'LiteLLM_DailyUserSpend'));
    observation_table CONSTANT REGCLASS := to_regclass(format('%I.%I', target_schema, 'LiteLLM_AutoRouterBaselineObservation'));
    -- This marker versions the full column/function/trigger contract, preserved by older pods.
    implementation_version CONSTANT INTEGER := 1;
    function_body CONSTANT TEXT := $function$
DECLARE
    previous_publication JSONB := OLD.daily_costs_publication::jsonb;
    next_publication JSONB := NEW.publication::jsonb;
    attribution JSONB := NEW.data::jsonb -> 'daily';
    previous_estimated BOOLEAN := COALESCE(previous_publication ->> 'status' = 'estimated', FALSE)
        AND previous_publication ->> 'actual_spend' IS NOT NULL
        AND previous_publication ->> 'baseline_spend' IS NOT NULL;
    next_estimated BOOLEAN := COALESCE(next_publication ->> 'status' = 'estimated', FALSE)
        AND next_publication ->> 'actual_spend' IS NOT NULL
        AND next_publication ->> 'baseline_spend' IS NOT NULL;
    request_delta BIGINT := (CASE WHEN next_estimated THEN 1 ELSE 0 END)
        - (CASE WHEN previous_estimated THEN 1 ELSE 0 END);
    actual_delta DOUBLE PRECISION :=
        (CASE WHEN next_estimated THEN (next_publication ->> 'actual_spend')::double precision ELSE 0 END)
        - (CASE WHEN previous_estimated THEN (previous_publication ->> 'actual_spend')::double precision ELSE 0 END);
    target_user_id TEXT;
BEGIN
    NEW.daily_costs_publication := NEW.publication;
    IF request_delta = 0 AND actual_delta = 0 THEN
        RETURN NEW;
    END IF;

    FOR target_user_id IN
        SELECT DISTINCT COALESCE(target ->> 'entity_id', '')
        FROM jsonb_array_elements(COALESCE(attribution -> 'targets', '[]'::jsonb)) AS target
        WHERE target ->> 'entity' = 'user'
        ORDER BY 1
    LOOP
        INSERT INTO "LiteLLM_DailyUserSpend" (
            id, user_id, date, api_key, model, custom_llm_provider,
            mcp_namespaced_tool_name, endpoint, model_group,
            autorouter_estimated_requests, autorouter_estimated_actual_spend, updated_at
        ) VALUES (
            'autorouter-coverage:' || length(NEW.request_id)::text || ':' || NEW.request_id || ':' || target_user_id,
            target_user_id, COALESCE(attribution ->> 'date', ''), COALESCE(attribution ->> 'api_key', ''),
            COALESCE(attribution ->> 'model', ''), COALESCE(attribution ->> 'custom_llm_provider', ''),
            COALESCE(attribution ->> 'mcp_namespaced_tool_name', ''), COALESCE(attribution ->> 'endpoint', ''),
            attribution ->> 'model_group', request_delta, actual_delta, (NOW() AT TIME ZONE 'UTC')
        )
        ON CONFLICT (user_id, date, api_key, model, custom_llm_provider, mcp_namespaced_tool_name, endpoint)
        DO UPDATE SET
            autorouter_estimated_requests = "LiteLLM_DailyUserSpend".autorouter_estimated_requests
                + EXCLUDED.autorouter_estimated_requests,
            autorouter_estimated_actual_spend = "LiteLLM_DailyUserSpend".autorouter_estimated_actual_spend
                + EXCLUDED.autorouter_estimated_actual_spend,
            updated_at = (NOW() AT TIME ZONE 'UTC');
    END LOOP;
    RETURN NEW;
END;
$function$;
    missing_columns BOOLEAN;
    invalid_columns BOOLEAN;
    function_id OID;
    function_version INTEGER;
    function_ready BOOLEAN;
    trigger_ready BOOLEAN;
BEGIN
    IF daily_table IS NULL OR observation_table IS NULL THEN
        RAISE EXCEPTION 'Daily auto-router coverage requires its tables in schema %', target_schema;
    END IF;

    FOR attempt IN 1..2 LOOP
        SELECT bool_or(a.attnum IS NULL),
            bool_or(a.attnum IS NOT NULL AND (
                a.atttypid <> required.column_type::regtype
                OR a.attnotnull <> required.not_null
                OR CASE WHEN required.not_null
                    THEN COALESCE(pg_get_expr(d.adbin, d.adrelid), '') !~ '^0([.]0+)?(::(bigint|double precision))?$'
                    ELSE d.oid IS NOT NULL END
            ))
        INTO missing_columns, invalid_columns
        FROM (VALUES
            (daily_table, 'autorouter_accounted_requests', 'bigint', TRUE),
            (daily_table, 'autorouter_requests', 'bigint', TRUE),
            (daily_table, 'autorouter_llm_spend', 'double precision', TRUE),
            (daily_table, 'autorouter_classifier_cost', 'double precision', TRUE),
            (daily_table, 'autorouter_classifier_cost_recorded_requests', 'bigint', TRUE),
            (daily_table, 'autorouter_estimated_requests', 'bigint', TRUE),
            (daily_table, 'autorouter_estimated_actual_spend', 'double precision', TRUE),
            (observation_table, 'daily_costs_publication', 'text', FALSE)
        ) AS required(table_id, column_name, column_type, not_null)
        LEFT JOIN pg_attribute a ON a.attrelid = required.table_id
            AND a.attname = required.column_name AND a.attnum > 0 AND NOT a.attisdropped
        LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum;

        IF invalid_columns THEN
            RAISE EXCEPTION 'Daily auto-router coverage has incompatible column definitions in schema %', target_schema;
        END IF;

        SELECT p.oid,
            substring(obj_description(p.oid, 'pg_proc') FROM '^litellm:autorouter_daily_coverage:([0-9]{1,9})$')::integer,
            p.prorettype = 'trigger'::regtype AND l.lanname = 'plpgsql'
                AND p.provolatile = 'v' AND NOT p.prosecdef AND p.proconfig IS NULL
                AND (p.prosrc = function_body OR COALESCE(
                    substring(obj_description(p.oid, 'pg_proc') FROM '^litellm:autorouter_daily_coverage:([0-9]{1,9})$')::integer,
                    0
                ) > implementation_version)
        INTO function_id, function_version, function_ready
        FROM pg_proc p
        JOIN pg_namespace n ON n.oid = p.pronamespace
        JOIN pg_language l ON l.oid = p.prolang
        WHERE n.nspname = target_schema AND p.proname = 'litellm_update_daily_autorouter_coverage'
            AND p.pronargs = 0;

        SELECT EXISTS (
            SELECT 1 FROM pg_trigger t
            JOIN pg_attribute a ON a.attrelid = observation_table AND a.attname = 'publication'
                AND a.attnum > 0 AND NOT a.attisdropped
            WHERE t.tgrelid = observation_table AND t.tgname = 'litellm_update_daily_autorouter_coverage'
                AND NOT t.tgisinternal AND t.tgenabled IN ('O', 'A') AND t.tgtype = 19
                AND t.tgattr::text = a.attnum::text AND t.tgnargs = 0 AND t.tgqual IS NULL
                AND t.tgfoid = function_id
        ) INTO trigger_ready;

        IF NOT missing_columns AND COALESCE(function_ready, FALSE) AND trigger_ready THEN
            RETURN;
        END IF;
        IF function_version > implementation_version THEN
            RAISE EXCEPTION 'Newer daily auto-router coverage is incompatible with this build in schema %', target_schema;
        END IF;
        IF attempt = 1 THEN
            PERFORM pg_advisory_xact_lock(1279874117, 1145129292);
        END IF;
    END LOOP;

    IF missing_columns THEN
        ALTER TABLE "LiteLLM_DailyUserSpend"
            ADD COLUMN IF NOT EXISTS "autorouter_accounted_requests" BIGINT NOT NULL DEFAULT 0,
            ADD COLUMN IF NOT EXISTS "autorouter_requests" BIGINT NOT NULL DEFAULT 0,
            ADD COLUMN IF NOT EXISTS "autorouter_llm_spend" DOUBLE PRECISION NOT NULL DEFAULT 0,
            ADD COLUMN IF NOT EXISTS "autorouter_classifier_cost" DOUBLE PRECISION NOT NULL DEFAULT 0,
            ADD COLUMN IF NOT EXISTS "autorouter_classifier_cost_recorded_requests" BIGINT NOT NULL DEFAULT 0,
            ADD COLUMN IF NOT EXISTS "autorouter_estimated_requests" BIGINT NOT NULL DEFAULT 0,
            ADD COLUMN IF NOT EXISTS "autorouter_estimated_actual_spend" DOUBLE PRECISION NOT NULL DEFAULT 0;

        ALTER TABLE "LiteLLM_AutoRouterBaselineObservation"
            ADD COLUMN IF NOT EXISTS "daily_costs_publication" TEXT;
    END IF;

    IF NOT COALESCE(function_ready, FALSE) THEN
        EXECUTE format(
            'CREATE OR REPLACE FUNCTION %I.litellm_update_daily_autorouter_coverage() '
            'RETURNS TRIGGER AS %L LANGUAGE plpgsql VOLATILE SECURITY INVOKER',
            target_schema, function_body
        );
        EXECUTE format('ALTER FUNCTION %I.litellm_update_daily_autorouter_coverage() RESET ALL', target_schema);
        EXECUTE format('COMMENT ON FUNCTION %I.litellm_update_daily_autorouter_coverage() IS %L',
            target_schema, 'litellm:autorouter_daily_coverage:' || implementation_version::text);
    END IF;

    IF NOT trigger_ready THEN
        DROP TRIGGER IF EXISTS litellm_update_daily_autorouter_coverage ON "LiteLLM_AutoRouterBaselineObservation";
        CREATE TRIGGER litellm_update_daily_autorouter_coverage
        BEFORE UPDATE OF publication ON "LiteLLM_AutoRouterBaselineObservation"
        FOR EACH ROW EXECUTE PROCEDURE litellm_update_daily_autorouter_coverage();
    END IF;
END;
$install$;

COMMIT;
