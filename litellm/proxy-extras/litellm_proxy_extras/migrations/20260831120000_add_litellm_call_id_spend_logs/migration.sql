-- AlterTable
ALTER TABLE "LiteLLM_SpendLogs" ADD COLUMN IF NOT EXISTS "litellm_call_id" TEXT;
