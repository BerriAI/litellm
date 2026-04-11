-- Add OAuth token validation fields to LiteLLM_MCPServerTable
-- AlterTable
ALTER TABLE "LiteLLM_MCPServerTable"
  ADD COLUMN IF NOT EXISTS "token_validation"          JSONB,
  ADD COLUMN IF NOT EXISTS "token_storage_ttl_seconds" INTEGER;
