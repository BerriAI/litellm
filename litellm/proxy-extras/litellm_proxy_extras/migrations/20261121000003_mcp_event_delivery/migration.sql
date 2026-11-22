SET LOCAL lock_timeout = '10s';

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
