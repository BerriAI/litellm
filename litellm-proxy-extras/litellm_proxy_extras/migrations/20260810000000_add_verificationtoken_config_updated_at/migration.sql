-- AlterTable
ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN IF NOT EXISTS "config_updated_at" TIMESTAMP(3);

-- AlterTable
ALTER TABLE "LiteLLM_DeletedVerificationToken" ADD COLUMN IF NOT EXISTS "config_updated_at" TIMESTAMP(3);
