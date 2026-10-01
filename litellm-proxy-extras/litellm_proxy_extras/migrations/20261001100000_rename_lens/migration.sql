DO $$
BEGIN
    ALTER TABLE IF EXISTS "LiteLLM_Engine" RENAME TO "LiteLLM_Lens";
    ALTER TABLE IF EXISTS "LiteLLM_EngineRun" RENAME TO "LiteLLM_LensRun";
    ALTER TABLE IF EXISTS "LiteLLM_EngineWorker" RENAME TO "LiteLLM_LensWorker";
    IF EXISTS (
        SELECT 1 FROM pg_attribute
        WHERE attrelid = to_regclass('"LiteLLM_LensRun"')
          AND attname = 'engine_id' AND NOT attisdropped
    ) THEN
        ALTER TABLE "LiteLLM_LensRun" RENAME COLUMN "engine_id" TO "lens_id";
    END IF;
    ALTER INDEX IF EXISTS "LiteLLM_Engine_pkey" RENAME TO "LiteLLM_Lens_pkey";
    ALTER INDEX IF EXISTS "LiteLLM_EngineRun_pkey" RENAME TO "LiteLLM_LensRun_pkey";
    ALTER INDEX IF EXISTS "LiteLLM_EngineWorker_pkey" RENAME TO "LiteLLM_LensWorker_pkey";
    ALTER INDEX IF EXISTS "LiteLLM_EngineWorker_token_hash_key" RENAME TO "LiteLLM_LensWorker_token_hash_key";
    ALTER INDEX IF EXISTS "LiteLLM_EngineRun_engine_id_created_at_idx" RENAME TO "LiteLLM_LensRun_lens_id_created_at_idx";
END $$;
