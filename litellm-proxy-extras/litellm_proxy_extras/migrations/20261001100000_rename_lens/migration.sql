ALTER TABLE "LiteLLM_Engine" RENAME TO "LiteLLM_Lens";
ALTER TABLE "LiteLLM_EngineRun" RENAME TO "LiteLLM_LensRun";
ALTER TABLE "LiteLLM_LensRun" RENAME COLUMN "engine_id" TO "lens_id";
ALTER TABLE "LiteLLM_EngineWorker" RENAME TO "LiteLLM_LensWorker";
ALTER TABLE "LiteLLM_Lens" RENAME CONSTRAINT "LiteLLM_Engine_pkey" TO "LiteLLM_Lens_pkey";
ALTER TABLE "LiteLLM_LensRun" RENAME CONSTRAINT "LiteLLM_EngineRun_pkey" TO "LiteLLM_LensRun_pkey";
ALTER TABLE "LiteLLM_LensWorker" RENAME CONSTRAINT "LiteLLM_EngineWorker_pkey" TO "LiteLLM_LensWorker_pkey";
ALTER TABLE "LiteLLM_LensWorker" RENAME CONSTRAINT "LiteLLM_EngineWorker_token_hash_key" TO "LiteLLM_LensWorker_token_hash_key";
ALTER INDEX "LiteLLM_EngineRun_engine_id_created_at_idx" RENAME TO "LiteLLM_LensRun_lens_id_created_at_idx";
