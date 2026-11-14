-- AlterTable
ALTER TABLE "LiteLLM_ManagedVectorStoresTable" ADD COLUMN IF NOT EXISTS "litellm_params" JSONB;

