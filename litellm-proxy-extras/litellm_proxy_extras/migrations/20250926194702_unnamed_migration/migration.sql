-- AlterTable
ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN     "auto_rotate" BOOLEAN DEFAULT false,
ADD COLUMN     "key_rotation_at" TIMESTAMP(3),
ADD COLUMN     "last_rotation_at" TIMESTAMP(3),
ADD COLUMN     "rotation_count" INTEGER DEFAULT 0,
ADD COLUMN     "rotation_interval" TEXT;

