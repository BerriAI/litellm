-- AlterTable
ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN IF NOT EXISTS "key_type" TEXT;

-- AlterTable
ALTER TABLE "LiteLLM_DeletedVerificationToken" ADD COLUMN IF NOT EXISTS "key_type" TEXT;
