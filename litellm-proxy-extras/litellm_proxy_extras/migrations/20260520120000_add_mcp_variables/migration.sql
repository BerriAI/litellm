-- AlterTable: add admin-configured variables to MCP server table
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN IF NOT EXISTS "variables" JSONB DEFAULT '[]';

-- CreateTable: per-user variable values for MCP servers
CREATE TABLE IF NOT EXISTS "LiteLLM_MCPUserVariables" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "server_id" TEXT NOT NULL,
    "values_b64" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_MCPUserVariables_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_MCPUserVariables_user_id_server_id_key" ON "LiteLLM_MCPUserVariables"("user_id", "server_id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_MCPUserVariables_user_id_idx" ON "LiteLLM_MCPUserVariables"("user_id");
