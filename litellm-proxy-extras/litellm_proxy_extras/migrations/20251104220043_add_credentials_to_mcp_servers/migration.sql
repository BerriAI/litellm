-- AlterTable
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN IF NOT EXISTS "credentials" JSONB DEFAULT '{}';
