SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_UserTable" ADD COLUMN IF NOT EXISTS "soft_budget" DOUBLE PRECISION;
