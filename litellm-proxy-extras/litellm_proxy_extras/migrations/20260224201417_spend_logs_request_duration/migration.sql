-- AlterTable
ALTER TABLE "LiteLLM_SpendLogs" ADD COLUMN IF NOT EXISTS "request_duration_ms" INTEGER;

