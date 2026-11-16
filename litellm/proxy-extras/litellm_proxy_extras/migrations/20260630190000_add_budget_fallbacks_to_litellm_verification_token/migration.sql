-- AlterTable
ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN IF NOT EXISTS "budget_fallbacks" JSONB NOT NULL DEFAULT '{}';

-- AlterTable
ALTER TABLE "LiteLLM_DeletedVerificationToken" ADD COLUMN IF NOT EXISTS "budget_fallbacks" JSONB NOT NULL DEFAULT '{}';
