-- AlterTable: add template/instance discriminator to MCP server table.
-- "instance" servers are live and loaded by the manager; "template" servers
-- are config blueprints only. template_id links an instance to its template.
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN IF NOT EXISTS "kind" TEXT NOT NULL DEFAULT 'instance';
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN IF NOT EXISTS "template_id" TEXT;
