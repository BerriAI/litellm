SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_TeamTable" ADD COLUMN IF NOT EXISTS "team_member_budget_limits" JSONB;
ALTER TABLE "LiteLLM_TeamMembership" ADD COLUMN IF NOT EXISTS "budget_limits" JSONB;
