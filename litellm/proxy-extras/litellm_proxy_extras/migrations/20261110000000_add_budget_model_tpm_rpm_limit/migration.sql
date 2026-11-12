SET LOCAL lock_timeout = '10s';

-- AlterTable
ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN IF NOT EXISTS "model_tpm_limit" JSONB;

-- AlterTable
ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN IF NOT EXISTS "model_rpm_limit" JSONB;
