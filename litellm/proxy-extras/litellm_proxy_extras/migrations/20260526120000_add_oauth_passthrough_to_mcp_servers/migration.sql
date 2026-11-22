-- AlterTable
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN IF NOT EXISTS "oauth_passthrough" BOOLEAN NOT NULL DEFAULT false;
