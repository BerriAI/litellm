-- AlterTable: Add new fields to LiteLLM_ProjectTable
ALTER TABLE "LiteLLM_ProjectTable" ADD COLUMN "description" TEXT;
ALTER TABLE "LiteLLM_ProjectTable" ADD COLUMN "model_rpm_limit" JSONB NOT NULL DEFAULT '{}';
ALTER TABLE "LiteLLM_ProjectTable" ADD COLUMN "model_tpm_limit" JSONB NOT NULL DEFAULT '{}';

