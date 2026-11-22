SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_ObjectPermissionTable" ADD COLUMN IF NOT EXISTS "apps" TEXT[] DEFAULT ARRAY[]::TEXT[];
