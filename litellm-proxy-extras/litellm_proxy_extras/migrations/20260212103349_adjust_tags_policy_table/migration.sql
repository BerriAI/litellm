-- AlterTable
ALTER TABLE "LiteLLM_PolicyAttachmentTable" ADD COLUMN IF NOT EXISTS "tags" TEXT[] DEFAULT ARRAY[]::TEXT[];

