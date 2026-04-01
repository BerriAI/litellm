-- AlterTable
ALTER TABLE "LiteLLM_PolicyAttachmentTable" ADD COLUMN     "tags" TEXT[] DEFAULT ARRAY[]::TEXT[];

