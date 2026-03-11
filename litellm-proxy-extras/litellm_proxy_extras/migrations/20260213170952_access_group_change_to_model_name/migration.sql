-- AlterTable
ALTER TABLE "LiteLLM_AccessGroupTable" DROP COLUMN "access_model_ids",
ADD COLUMN     "access_model_names" TEXT[] DEFAULT ARRAY[]::TEXT[];
