SET LOCAL lock_timeout = '10s';
ALTER TABLE "LiteLLM_TeamTable" ADD COLUMN IF NOT EXISTS "cost_margin_percent" DOUBLE PRECISION;
