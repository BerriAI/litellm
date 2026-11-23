SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_MCPEventSubscription"
ADD COLUMN IF NOT EXISTS "generation" TEXT NOT NULL DEFAULT 'initial';
CREATE SEQUENCE IF NOT EXISTS mcp_event_generation_seq
OWNED BY "LiteLLM_MCPEventSubscription"."generation";

ALTER TABLE "LiteLLM_MCPEventSubscription"
ALTER COLUMN "generation" SET DEFAULT ('db-' || nextval('mcp_event_generation_seq')::text);
