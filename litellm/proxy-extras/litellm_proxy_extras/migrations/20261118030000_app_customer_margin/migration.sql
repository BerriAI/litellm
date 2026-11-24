SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_AppTable"
  ADD COLUMN IF NOT EXISTS "customer_margin" DOUBLE PRECISION;
