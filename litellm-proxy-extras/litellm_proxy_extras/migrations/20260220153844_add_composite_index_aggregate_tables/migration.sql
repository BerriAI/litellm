-- DropIndex
DROP INDEX IF EXISTS "LiteLLM_DailyAgentSpend_agent_id_idx";

-- DropIndex
DROP INDEX IF EXISTS "LiteLLM_DailyEndUserSpend_end_user_id_idx";

-- DropIndex
DROP INDEX IF EXISTS "LiteLLM_DailyOrganizationSpend_organization_id_idx";

-- DropIndex
DROP INDEX IF EXISTS "LiteLLM_DailyTagSpend_tag_idx";

-- DropIndex
DROP INDEX IF EXISTS "LiteLLM_DailyTeamSpend_team_id_idx";

-- DropIndex
DROP INDEX IF EXISTS "LiteLLM_DailyUserSpend_user_id_idx";

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_DailyAgentSpend_agent_id_date_idx" ON "LiteLLM_DailyAgentSpend"("agent_id", "date");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_DailyEndUserSpend_end_user_id_date_idx" ON "LiteLLM_DailyEndUserSpend"("end_user_id", "date");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_DailyOrganizationSpend_organization_id_date_idx" ON "LiteLLM_DailyOrganizationSpend"("organization_id", "date");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_DailyTagSpend_tag_date_idx" ON "LiteLLM_DailyTagSpend"("tag", "date");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_DailyTeamSpend_team_id_date_idx" ON "LiteLLM_DailyTeamSpend"("team_id", "date");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_DailyUserSpend_user_id_date_idx" ON "LiteLLM_DailyUserSpend"("user_id", "date");

