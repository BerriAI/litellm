DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_attribute
        WHERE attrelid = to_regclass('"LiteLLM_AutoRouterDailySpend"')
          AND attname = 'total_tokens' AND NOT attisdropped
    ) THEN
        ALTER TABLE "LiteLLM_AutoRouterDailySpend"
            ADD COLUMN IF NOT EXISTS "total_tokens" BIGINT NOT NULL DEFAULT 0;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_attribute
        WHERE attrelid = to_regclass('"LiteLLM_AutoRouterDailySpend"')
          AND attname = 'token_recorded_turns' AND NOT attisdropped
    ) THEN
        ALTER TABLE "LiteLLM_AutoRouterDailySpend"
            ADD COLUMN IF NOT EXISTS "token_recorded_turns" INTEGER NOT NULL DEFAULT 0;
    END IF;
END $$;
