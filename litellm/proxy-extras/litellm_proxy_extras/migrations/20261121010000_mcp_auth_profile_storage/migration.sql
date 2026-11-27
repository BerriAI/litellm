SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN IF NOT EXISTS "additional_auth" JSONB;
ALTER TABLE "LiteLLM_ObjectPermissionTable" ADD COLUMN IF NOT EXISTS "mcp_auth_profiles" JSONB;
