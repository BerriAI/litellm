SET LOCAL lock_timeout = '10s';
CREATE TABLE IF NOT EXISTS "LiteLLM_SpendLogDelivery" (
    "delivery_id" TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    "request_id" TEXT NOT NULL,
    "state" "AccountingDeliveryState" NOT NULL DEFAULT 'pending',
    "attempts" INTEGER NOT NULL DEFAULT 0,
    "next_attempt_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "applied_at" TIMESTAMP(3),
    "last_error" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "accounting_payload" JSONB NOT NULL,
    "is_batch" BOOLEAN NOT NULL
);
CREATE INDEX IF NOT EXISTS "LiteLLM_SpendLogDelivery_state_next_attempt_at_idx"
    ON "LiteLLM_SpendLogDelivery" ("state", "next_attempt_at");
CREATE INDEX IF NOT EXISTS "LiteLLM_SpendLogDelivery_state_applied_at_idx"
    ON "LiteLLM_SpendLogDelivery" ("state", "applied_at");
CREATE INDEX IF NOT EXISTS "LiteLLM_SpendLogDelivery_request_id_idx"
    ON "LiteLLM_SpendLogDelivery" ("request_id");
