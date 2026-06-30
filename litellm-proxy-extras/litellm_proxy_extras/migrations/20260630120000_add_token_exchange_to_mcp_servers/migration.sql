-- AlterTable
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN IF NOT EXISTS "token_exchange_endpoint" TEXT;
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN IF NOT EXISTS "audience" TEXT;
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN IF NOT EXISTS "subject_token_type" TEXT;
