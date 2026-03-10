-- AlterTable
ALTER TABLE "LiteLLM_AgentsTable" ADD COLUMN "tpm_limit" INTEGER;
ALTER TABLE "LiteLLM_AgentsTable" ADD COLUMN "rpm_limit" INTEGER;
ALTER TABLE "LiteLLM_AgentsTable" ADD COLUMN "session_tpm_limit" INTEGER;
ALTER TABLE "LiteLLM_AgentsTable" ADD COLUMN "session_rpm_limit" INTEGER;
