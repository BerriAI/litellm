SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_ManagedObjectTable" ADD COLUMN IF NOT EXISTS "creator_billing_context" JSONB;
