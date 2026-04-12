-- AlterTable
ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN IF NOT EXISTS "auto_rotate" BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS "key_rotation_at" TIMESTAMP(3),
ADD COLUMN IF NOT EXISTS "last_rotation_at" TIMESTAMP(3),
ADD COLUMN IF NOT EXISTS "rotation_count" INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS "rotation_interval" TEXT;

