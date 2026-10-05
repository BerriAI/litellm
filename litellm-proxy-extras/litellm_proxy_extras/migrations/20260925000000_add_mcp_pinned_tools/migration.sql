-- AlterTable
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN IF NOT EXISTS "pinned_tools" JSONB DEFAULT '{}';
