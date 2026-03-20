-- AlterTable
ALTER TABLE "LiteLLM_AgentsTable" ADD COLUMN IF NOT EXISTS "tpm_limit" INTEGER;
ALTER TABLE "LiteLLM_AgentsTable" ADD COLUMN IF NOT EXISTS "rpm_limit" INTEGER;
ALTER TABLE "LiteLLM_AgentsTable" ADD COLUMN IF NOT EXISTS "session_tpm_limit" INTEGER;
ALTER TABLE "LiteLLM_AgentsTable" ADD COLUMN IF NOT EXISTS "session_rpm_limit" INTEGER;
