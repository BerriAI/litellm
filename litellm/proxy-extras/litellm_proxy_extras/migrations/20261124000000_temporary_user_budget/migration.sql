SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_UserTable"
    ADD COLUMN IF NOT EXISTS "temp_budget_increase" DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS "temp_budget_expiry" TIMESTAMP(3);
