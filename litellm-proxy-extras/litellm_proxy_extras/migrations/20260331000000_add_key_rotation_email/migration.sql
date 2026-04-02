-- Add key_rotation_email to LiteLLM_VerificationToken and LiteLLM_DeletedVerificationToken
-- AlterTable
ALTER TABLE "LiteLLM_VerificationToken"
  ADD COLUMN IF NOT EXISTS "key_rotation_email" TEXT;

ALTER TABLE "LiteLLM_DeletedVerificationToken"
  ADD COLUMN IF NOT EXISTS "key_rotation_email" TEXT;
