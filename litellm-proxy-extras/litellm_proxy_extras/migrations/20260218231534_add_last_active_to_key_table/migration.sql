-- AlterTable
ALTER TABLE "LiteLLM_DeletedVerificationToken" ADD COLUMN     "last_active" TIMESTAMP(3);

-- AlterTable
ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN     "last_active" TIMESTAMP(3);

