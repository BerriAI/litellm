-- AlterTable
ALTER TABLE "LiteLLM_ObjectPermissionTable" ADD COLUMN     "vector_stores" TEXT[] DEFAULT ARRAY[]::TEXT[];

