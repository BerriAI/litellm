-- AlterTable
ALTER TABLE "LiteLLM_ObjectPermissionTable" ADD COLUMN IF NOT EXISTS "mcp_tool_approved_tools" JSONB;
