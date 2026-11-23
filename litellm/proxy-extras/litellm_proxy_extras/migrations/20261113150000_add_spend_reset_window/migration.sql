SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN IF NOT EXISTS "budget_reset_count" INTEGER NOT NULL DEFAULT 0;

ALTER TABLE "LiteLLM_EndUserTable" ADD COLUMN IF NOT EXISTS "spend_reset_window" TEXT;

ALTER TABLE "LiteLLM_CustomerTeamTable" ADD COLUMN IF NOT EXISTS "spend_reset_window" TEXT;

ALTER TABLE "LiteLLM_CustomerMemberTable" ADD COLUMN IF NOT EXISTS "spend_reset_window" TEXT;
