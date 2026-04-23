-- AlterTable
ALTER TABLE "LiteLLM_ObjectPermissionTable" ADD COLUMN IF NOT EXISTS "vector_stores" TEXT[] DEFAULT ARRAY[]::TEXT[];

