-- AlterTable
ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN IF NOT EXISTS "budget_reset_alignment" TEXT;

-- AlterTable
ALTER TABLE "LiteLLM_DeletedVerificationToken" ADD COLUMN IF NOT EXISTS "budget_reset_alignment" TEXT;

-- AlterTable
ALTER TABLE "LiteLLM_TeamTable" ADD COLUMN IF NOT EXISTS "budget_reset_alignment" TEXT;

-- AlterTable
ALTER TABLE "LiteLLM_DeletedTeamTable" ADD COLUMN IF NOT EXISTS "budget_reset_alignment" TEXT;

-- AlterTable
ALTER TABLE "LiteLLM_UserTable" ADD COLUMN IF NOT EXISTS "budget_reset_alignment" TEXT;
