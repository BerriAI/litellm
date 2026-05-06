-- AlterTable
ALTER TABLE "LiteLLM_AccessGroupTable"
ADD COLUMN IF NOT EXISTS "access_pass_through_routes" TEXT[] DEFAULT ARRAY[]::TEXT[],
ADD COLUMN IF NOT EXISTS "access_vector_store_ids" TEXT[] DEFAULT ARRAY[]::TEXT[];
