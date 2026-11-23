SET LOCAL lock_timeout = '10s';

CREATE TABLE IF NOT EXISTS "LiteLLM_MCPEventSubscription" (
    "id" TEXT NOT NULL,
    "principal" TEXT NOT NULL,
    "key_hash" TEXT,
    "user_id" TEXT,
    "team_id" TEXT,
    "server_id" TEXT NOT NULL,
    "event_name" TEXT NOT NULL,
    "arguments" JSONB NOT NULL DEFAULT '{}',
    "client_url" TEXT NOT NULL,
    "client_secret" TEXT NOT NULL,
    "upstream_subscription_id" TEXT,
    "upstream_secret" TEXT NOT NULL,
    "upstream_refresh_before" TIMESTAMP(3),
    "refresh_before" TIMESTAMP(3),
    "cursor" TEXT,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "verified_at" TIMESTAMP(3),
    "last_delivery_at" TIMESTAMP(3),
    "last_error" TEXT,
    "failed_since" TIMESTAMP(3),
    "consecutive_failures" INTEGER NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_MCPEventSubscription_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "LiteLLM_MCPEventSubscription_principal_idx" ON "LiteLLM_MCPEventSubscription"("principal");
CREATE INDEX IF NOT EXISTS "LiteLLM_MCPEventSubscription_server_id_idx" ON "LiteLLM_MCPEventSubscription"("server_id");
CREATE INDEX IF NOT EXISTS "LiteLLM_MCPEventSubscription_key_hash_idx" ON "LiteLLM_MCPEventSubscription"("key_hash");
CREATE INDEX IF NOT EXISTS "LiteLLM_MCPEventSubscription_status_idx" ON "LiteLLM_MCPEventSubscription"("status");
