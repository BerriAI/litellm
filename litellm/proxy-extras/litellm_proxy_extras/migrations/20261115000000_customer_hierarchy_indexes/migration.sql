SET LOCAL lock_timeout = '10s';

CREATE INDEX IF NOT EXISTS "LiteLLM_CustomerMemberTable_parent_team_ids_idx" ON "LiteLLM_CustomerMemberTable" USING GIN ("parent_team_ids");
CREATE INDEX IF NOT EXISTS "LiteLLM_CustomerMemberTable_budget_id_idx" ON "LiteLLM_CustomerMemberTable"("budget_id");
CREATE INDEX IF NOT EXISTS "LiteLLM_CustomerMemberTable_budget_override_id_idx" ON "LiteLLM_CustomerMemberTable"("budget_override_id");
CREATE INDEX IF NOT EXISTS "LiteLLM_CustomerTeamTable_budget_id_idx" ON "LiteLLM_CustomerTeamTable"("budget_id");
CREATE INDEX IF NOT EXISTS "LiteLLM_CustomerTeamTable_budget_override_id_idx" ON "LiteLLM_CustomerTeamTable"("budget_override_id");
