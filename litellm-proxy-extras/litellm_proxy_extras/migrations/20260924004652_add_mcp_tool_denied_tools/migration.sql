-- AlterTable
ALTER TABLE "LiteLLM_ObjectPermissionTable" ADD COLUMN IF NOT EXISTS "mcp_tool_denied_tools" JSONB;
