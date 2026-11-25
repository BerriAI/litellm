SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN IF NOT EXISTS "events" JSONB;
