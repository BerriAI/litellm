-- AlterTable: add budget_limits column to LiteLLM_VerificationToken
ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN IF NOT EXISTS "budget_limits" JSONB;

-- AlterTable: add budget_limits column to LiteLLM_TeamTable
ALTER TABLE "LiteLLM_TeamTable" ADD COLUMN IF NOT EXISTS "budget_limits" JSONB;
