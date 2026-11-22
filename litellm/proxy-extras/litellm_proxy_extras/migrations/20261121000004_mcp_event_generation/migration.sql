SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_MCPEventSubscription"
ADD COLUMN IF NOT EXISTS "generation" TEXT NOT NULL DEFAULT gen_random_uuid()::text;
