-- AlterTable: Add source_url field to LiteLLM_MCPServerTable for GitHub/docs link
ALTER TABLE "LiteLLM_MCPServerTable"
  ADD COLUMN IF NOT EXISTS "source_url" TEXT;
