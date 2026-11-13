SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN IF NOT EXISTS "model_limit_max_borrow_multiple" DOUBLE PRECISION;
