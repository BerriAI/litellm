-- AlterTable
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN IF NOT EXISTS "allowed_cidrs" TEXT[] DEFAULT ARRAY[]::TEXT[];
