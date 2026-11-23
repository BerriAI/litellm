-- AlterTable
ALTER TABLE "LiteLLM_ObjectPermissionTable" ADD COLUMN IF NOT EXISTS "models" TEXT[] DEFAULT ARRAY[]::TEXT[];

