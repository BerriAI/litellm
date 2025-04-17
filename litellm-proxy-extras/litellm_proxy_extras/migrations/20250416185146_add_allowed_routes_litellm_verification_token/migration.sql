-- AlterTable
ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN     "allowed_routes" TEXT[] DEFAULT ARRAY[]::TEXT[];

