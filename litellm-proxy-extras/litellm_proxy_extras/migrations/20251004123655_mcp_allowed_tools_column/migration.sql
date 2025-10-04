-- AlterTable
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN     "allowed_tools" TEXT[] DEFAULT ARRAY[]::TEXT[];

