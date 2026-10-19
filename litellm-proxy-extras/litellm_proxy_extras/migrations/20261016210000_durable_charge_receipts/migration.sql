SET LOCAL lock_timeout = '10s';

CREATE TABLE IF NOT EXISTS "LiteLLM_ChargeReceipt" (
    "event_id" TEXT NOT NULL,
    "key_hash" TEXT NOT NULL,
    "payload" JSONB NOT NULL,
    "payload_hash" TEXT NOT NULL,
    "key_counter_updated" BOOLEAN NOT NULL,
    "created_at" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "retry_at" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "attempts" INTEGER NOT NULL DEFAULT 0,
    "lease_owner" TEXT,
    "lease_until" TIMESTAMPTZ(3),
    "delivered_at" TIMESTAMPTZ(3),
    "last_error" TEXT,
    CONSTRAINT "LiteLLM_ChargeReceipt_pkey" PRIMARY KEY ("event_id")
);

CREATE INDEX IF NOT EXISTS "LiteLLM_ChargeReceipt_pending_idx"
ON "LiteLLM_ChargeReceipt" ("delivered_at", "retry_at", "lease_until", "event_id");

CREATE INDEX IF NOT EXISTS "LiteLLM_ChargeReceipt_key_idx"
ON "LiteLLM_ChargeReceipt" ("key_hash", "created_at");
