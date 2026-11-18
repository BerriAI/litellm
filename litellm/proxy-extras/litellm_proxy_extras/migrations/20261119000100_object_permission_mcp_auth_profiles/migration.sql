SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_ObjectPermissionTable" ADD COLUMN IF NOT EXISTS "mcp_auth_profiles" TEXT[] DEFAULT ARRAY[]::TEXT[];
