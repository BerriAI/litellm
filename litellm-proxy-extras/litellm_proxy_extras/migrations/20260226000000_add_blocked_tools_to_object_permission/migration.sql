-- AlterTable
ALTER TABLE "LiteLLM_ObjectPermissionTable" ADD COLUMN IF NOT EXISTS "blocked_tools" TEXT[] DEFAULT ARRAY[]::TEXT[];
