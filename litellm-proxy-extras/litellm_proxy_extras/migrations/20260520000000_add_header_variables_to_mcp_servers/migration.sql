-- AlterTable
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN IF NOT EXISTS "header_variables" JSONB DEFAULT '[]';
