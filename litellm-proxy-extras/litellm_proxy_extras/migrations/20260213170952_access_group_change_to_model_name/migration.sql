-- AlterTable
ALTER TABLE "LiteLLM_AccessGroupTable" DROP COLUMN IF EXISTS "access_model_ids",
ADD COLUMN IF NOT EXISTS "access_model_names" TEXT[] DEFAULT ARRAY[]::TEXT[];
