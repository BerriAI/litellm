SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_AppTable"
  ADD COLUMN IF NOT EXISTS "auto_create_customers" BOOLEAN NOT NULL DEFAULT true;
