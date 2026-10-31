ALTER TABLE "LiteLLM_TeamTable" ADD COLUMN "team_member_budget_limits" JSONB;
ALTER TABLE "LiteLLM_TeamMembership" ADD COLUMN "budget_limits" JSONB;
