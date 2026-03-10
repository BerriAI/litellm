-- AlterTable
ALTER TABLE "LiteLLM_ObjectPermissionTable" ADD COLUMN "blocked_tools" TEXT[] DEFAULT ARRAY[]::TEXT[];
