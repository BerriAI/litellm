-- AlterTable: Add ITPM/OTPM fields to LiteLLM_ProjectTable
ALTER TABLE "LiteLLM_ProjectTable" ADD COLUMN IF NOT EXISTS "model_itpm_limit" JSONB NOT NULL DEFAULT '{}';
ALTER TABLE "LiteLLM_ProjectTable" ADD COLUMN IF NOT EXISTS "model_otpm_limit" JSONB NOT NULL DEFAULT '{}';
