-- Convert `date` column on daily-aggregate spend / metrics tables from TEXT (YYYY-MM-DD)
-- to native PostgreSQL DATE. The text column has always been a date-of-day in YYYY-MM-DD
-- format, so the cast is lossless. This unlocks proper range queries / sorts /
-- date arithmetic without `::date` casting (which prevented index usage), and brings the
-- column type in line with what it actually represents.
--
-- Notes:
-- - Each ALTER COLUMN is wrapped in a guard against repeated runs (column already DATE).
-- - All affected unique constraints / indexes implicitly retain their definitions; the
--   underlying column type changes but indexes are rebuilt by Postgres automatically.

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'LiteLLM_DailyUserSpend' AND column_name = 'date' AND data_type = 'text'
    ) THEN
        ALTER TABLE "LiteLLM_DailyUserSpend"
            ALTER COLUMN "date" TYPE DATE USING "date"::date;
    END IF;
END $$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'LiteLLM_DailyOrganizationSpend' AND column_name = 'date' AND data_type = 'text'
    ) THEN
        ALTER TABLE "LiteLLM_DailyOrganizationSpend"
            ALTER COLUMN "date" TYPE DATE USING "date"::date;
    END IF;
END $$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'LiteLLM_DailyEndUserSpend' AND column_name = 'date' AND data_type = 'text'
    ) THEN
        ALTER TABLE "LiteLLM_DailyEndUserSpend"
            ALTER COLUMN "date" TYPE DATE USING "date"::date;
    END IF;
END $$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'LiteLLM_DailyAgentSpend' AND column_name = 'date' AND data_type = 'text'
    ) THEN
        ALTER TABLE "LiteLLM_DailyAgentSpend"
            ALTER COLUMN "date" TYPE DATE USING "date"::date;
    END IF;
END $$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'LiteLLM_DailyTeamSpend' AND column_name = 'date' AND data_type = 'text'
    ) THEN
        ALTER TABLE "LiteLLM_DailyTeamSpend"
            ALTER COLUMN "date" TYPE DATE USING "date"::date;
    END IF;
END $$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'LiteLLM_DailyTagSpend' AND column_name = 'date' AND data_type = 'text'
    ) THEN
        ALTER TABLE "LiteLLM_DailyTagSpend"
            ALTER COLUMN "date" TYPE DATE USING "date"::date;
    END IF;
END $$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'LiteLLM_DailyGuardrailMetrics' AND column_name = 'date' AND data_type = 'text'
    ) THEN
        ALTER TABLE "LiteLLM_DailyGuardrailMetrics"
            ALTER COLUMN "date" TYPE DATE USING "date"::date;
    END IF;
END $$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'LiteLLM_DailyPolicyMetrics' AND column_name = 'date' AND data_type = 'text'
    ) THEN
        ALTER TABLE "LiteLLM_DailyPolicyMetrics"
            ALTER COLUMN "date" TYPE DATE USING "date"::date;
    END IF;
END $$;
