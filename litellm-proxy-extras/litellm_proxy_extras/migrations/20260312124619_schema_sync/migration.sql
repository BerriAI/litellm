-- AlterTable
ALTER TABLE "LiteLLM_ObjectPermissionTable" ADD COLUMN     "models" TEXT[] DEFAULT ARRAY[]::TEXT[];

