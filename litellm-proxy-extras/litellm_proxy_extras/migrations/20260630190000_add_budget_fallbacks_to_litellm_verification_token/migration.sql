-- AlterTable
ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN IF NOT EXISTS "budget_fallbacks" JSONB NOT NULL DEFAULT '{}';
