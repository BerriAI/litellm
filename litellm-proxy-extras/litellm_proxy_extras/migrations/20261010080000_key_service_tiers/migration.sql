ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN "allowed_service_tiers" TEXT[] DEFAULT ARRAY[]::TEXT[];
ALTER TABLE "LiteLLM_DeletedVerificationToken" ADD COLUMN "allowed_service_tiers" TEXT[] DEFAULT ARRAY[]::TEXT[];
