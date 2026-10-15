SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN IF NOT EXISTS "model_throttling" JSONB;
ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN IF NOT EXISTS "model_throttling" JSONB;
ALTER TABLE "LiteLLM_UserTable" ADD COLUMN IF NOT EXISTS "model_throttling" JSONB;
ALTER TABLE "LiteLLM_TeamTable" ADD COLUMN IF NOT EXISTS "model_throttling" JSONB;
ALTER TABLE "LiteLLM_DeletedVerificationToken" ADD COLUMN IF NOT EXISTS "model_throttling" JSONB;
ALTER TABLE "LiteLLM_DeletedTeamTable" ADD COLUMN IF NOT EXISTS "model_throttling" JSONB;
