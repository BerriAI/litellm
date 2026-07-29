-- Add per-member model scope to LiteLLM_BudgetTable
-- allowed_models: empty array = inherit team models; non-empty = enforce member-level restriction
ALTER TABLE "LiteLLM_BudgetTable"
  ADD COLUMN IF NOT EXISTS "allowed_models" TEXT[] DEFAULT ARRAY[]::TEXT[];

-- Add default_team_member_models to LiteLLM_TeamTable
-- Seeds allowed_models for newly added team members; empty = no per-member restriction
ALTER TABLE "LiteLLM_TeamTable"
  ADD COLUMN IF NOT EXISTS "default_team_member_models" TEXT[] DEFAULT ARRAY[]::TEXT[];
