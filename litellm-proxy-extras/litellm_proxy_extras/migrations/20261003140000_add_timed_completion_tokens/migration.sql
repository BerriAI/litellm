ALTER TABLE "LiteLLM_DailyUserSpend" ADD COLUMN IF NOT EXISTS "timed_completion_tokens" BIGINT;
ALTER TABLE "LiteLLM_DailyOrganizationSpend" ADD COLUMN IF NOT EXISTS "timed_completion_tokens" BIGINT;
ALTER TABLE "LiteLLM_DailyEndUserSpend" ADD COLUMN IF NOT EXISTS "timed_completion_tokens" BIGINT;
ALTER TABLE "LiteLLM_DailyAgentSpend" ADD COLUMN IF NOT EXISTS "timed_completion_tokens" BIGINT;
ALTER TABLE "LiteLLM_DailyTeamSpend" ADD COLUMN IF NOT EXISTS "timed_completion_tokens" BIGINT;
ALTER TABLE "LiteLLM_DailyTagSpend" ADD COLUMN IF NOT EXISTS "timed_completion_tokens" BIGINT;
