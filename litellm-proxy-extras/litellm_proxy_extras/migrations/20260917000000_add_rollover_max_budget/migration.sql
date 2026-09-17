-- AlterTable
ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN IF NOT EXISTS "rollover_max_budget" DOUBLE PRECISION;

-- AlterTable
ALTER TABLE "LiteLLM_TeamTable" ADD COLUMN IF NOT EXISTS "rollover_max_budget" DOUBLE PRECISION;

-- AlterTable
ALTER TABLE "LiteLLM_UserTable" ADD COLUMN IF NOT EXISTS "rollover_max_budget" DOUBLE PRECISION;
