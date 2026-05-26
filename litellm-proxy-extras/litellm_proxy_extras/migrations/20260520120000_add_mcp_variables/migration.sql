-- AlterTable: add admin-configured variables to MCP server table
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN IF NOT EXISTS "variables" JSONB DEFAULT '[]';

-- CreateTable: per-user variable values, shared globally across all MCP servers
-- (one row per user; values_b64 is an encrypted JSON object {VAR_NAME: "value"}).
CREATE TABLE IF NOT EXISTS "LiteLLM_MCPUserVariables" (
    "user_id" TEXT NOT NULL,
    "values_b64" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_MCPUserVariables_pkey" PRIMARY KEY ("user_id")
);
