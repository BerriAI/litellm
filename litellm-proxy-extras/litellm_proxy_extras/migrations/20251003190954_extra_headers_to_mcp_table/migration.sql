-- AlterTable
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN     "extra_headers" TEXT[] DEFAULT ARRAY[]::TEXT[];

