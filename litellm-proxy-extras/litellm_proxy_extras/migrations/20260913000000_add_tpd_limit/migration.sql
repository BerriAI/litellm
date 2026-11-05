-- AlterTable
ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN IF NOT EXISTS "tpd_limit" BIGINT;

-- AlterTable
ALTER TABLE "LiteLLM_TeamTable" ADD COLUMN IF NOT EXISTS "tpd_limit" BIGINT;

-- AlterTable
ALTER TABLE "LiteLLM_DeletedTeamTable" ADD COLUMN IF NOT EXISTS "tpd_limit" BIGINT;

-- AlterTable
ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN IF NOT EXISTS "tpd_limit" BIGINT;

-- AlterTable
ALTER TABLE "LiteLLM_DeletedVerificationToken" ADD COLUMN IF NOT EXISTS "tpd_limit" BIGINT;
