ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN "model_throttling" JSONB;
ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN "model_throttling" JSONB;
ALTER TABLE "LiteLLM_UserTable" ADD COLUMN "model_throttling" JSONB;
ALTER TABLE "LiteLLM_TeamTable" ADD COLUMN "model_throttling" JSONB;
ALTER TABLE "LiteLLM_DeletedVerificationToken" ADD COLUMN "model_throttling" JSONB;
ALTER TABLE "LiteLLM_DeletedTeamTable" ADD COLUMN "model_throttling" JSONB;
