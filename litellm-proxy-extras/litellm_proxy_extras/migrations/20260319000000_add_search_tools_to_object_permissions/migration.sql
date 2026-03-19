-- AlterTable
ALTER TABLE "LiteLLM_ObjectPermissionTable" ADD COLUMN     "search_tools" TEXT[] DEFAULT ARRAY[]::TEXT[];
