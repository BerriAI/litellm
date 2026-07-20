-- AlterTable
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN IF NOT EXISTS "allowed_response_headers" TEXT[] DEFAULT ARRAY[]::TEXT[];
