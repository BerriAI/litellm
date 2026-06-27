-- AlterTable
ALTER TABLE "LiteLLM_AccessGroupTable" ADD COLUMN     "access_passthrough_routes" TEXT[] DEFAULT ARRAY[]::TEXT[],
ADD COLUMN     "access_vector_store_ids" TEXT[] DEFAULT ARRAY[]::TEXT[];

