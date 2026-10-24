SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN IF NOT EXISTS "tools_ttl_ms" INTEGER;
ALTER TABLE "LiteLLM_MCPToolsetTable" ADD COLUMN IF NOT EXISTS "tools_ttl_ms" INTEGER;
