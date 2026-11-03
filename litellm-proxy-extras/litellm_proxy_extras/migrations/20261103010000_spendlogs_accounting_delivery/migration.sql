SET LOCAL lock_timeout = '10s';
ALTER TABLE "LiteLLM_SpendLogs" ADD COLUMN IF NOT EXISTS "accounting_path" TEXT;
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = '"LiteLLM_SpendLogs"'::regclass
          AND conname = 'LiteLLM_SpendLogs_accounting_path_check'
    ) THEN
        ALTER TABLE "LiteLLM_SpendLogs" ADD CONSTRAINT "LiteLLM_SpendLogs_accounting_path_check"
            CHECK (accounting_path IS NULL OR accounting_path IN ('legacy', 'ledger'));
    END IF;
END $$;
CREATE TABLE IF NOT EXISTS "LiteLLM_SpendLogDelivery" (
    "request_id" TEXT PRIMARY KEY,
    "state" "AccountingDeliveryState" NOT NULL DEFAULT 'pending',
    "attempts" INTEGER NOT NULL DEFAULT 0,
    "next_attempt_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "applied_at" TIMESTAMP(3),
    "last_error" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "accounting_payload" JSONB NOT NULL,
    "previous_payload" JSONB,
    "is_batch" BOOLEAN NOT NULL
);
CREATE INDEX IF NOT EXISTS "LiteLLM_SpendLogDelivery_state_next_attempt_at_idx"
    ON "LiteLLM_SpendLogDelivery" ("state", "next_attempt_at");
CREATE INDEX IF NOT EXISTS "LiteLLM_SpendLogDelivery_state_applied_at_idx"
    ON "LiteLLM_SpendLogDelivery" ("state", "applied_at");
