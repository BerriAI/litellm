SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_ObjectPermissionTable" ADD COLUMN IF NOT EXISTS "mcp_event_permissions" JSONB;

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

CREATE TABLE IF NOT EXISTS "LiteLLM_MCPEventDelivery" (
    "delivery_id" TEXT NOT NULL,
    "subscription_id" TEXT NOT NULL,
    "body" TEXT NOT NULL,
    "webhook_id" TEXT NOT NULL,
    "attempt" INTEGER NOT NULL DEFAULT 0,
    "next_attempt_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "claimed_until" TIMESTAMP(3),
    "claim_id" TEXT,
    "prepared" BOOLEAN NOT NULL DEFAULT false,
    "delivered_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_MCPEventDelivery_pkey" PRIMARY KEY ("delivery_id")
);

CREATE INDEX IF NOT EXISTS "LiteLLM_MCPEventDelivery_next_attempt_at_idx" ON "LiteLLM_MCPEventDelivery"("next_attempt_at");
CREATE INDEX IF NOT EXISTS "LiteLLM_MCPEventDelivery_subscription_id_idx" ON "LiteLLM_MCPEventDelivery"("subscription_id");
CREATE INDEX IF NOT EXISTS "LiteLLM_MCPEventDelivery_delivered_at_idx" ON "LiteLLM_MCPEventDelivery"("delivered_at");
