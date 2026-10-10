ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN "allowed_service_tiers" JSONB;
ALTER TABLE "LiteLLM_DeletedVerificationToken" ADD COLUMN "allowed_service_tiers" JSONB;
