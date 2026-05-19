-- AlterTable: add budget_limits column to LiteLLM_UserTable
ALTER TABLE "LiteLLM_UserTable" ADD COLUMN IF NOT EXISTS "budget_limits" JSONB;
