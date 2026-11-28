SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN IF NOT EXISTS "oauth_identity_binding" JSONB;
