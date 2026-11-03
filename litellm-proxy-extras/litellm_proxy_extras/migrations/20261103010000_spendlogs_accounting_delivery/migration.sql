BEGIN;
SET LOCAL lock_timeout = '10s';
DO $$
BEGIN
    -- Only replace the old event-target receipts. A replay must retain all
    -- request-ledger receipts, including pending and parked work.
    IF EXISTS (
        SELECT 1 FROM pg_attribute
        WHERE attrelid = to_regclass('"LiteLLM_AccountingDelivery"')
          AND attname = 'event_id' AND NOT attisdropped
    ) THEN
        LOCK TABLE "LiteLLM_AccountingDelivery" IN ACCESS EXCLUSIVE MODE;
        IF EXISTS (SELECT 1 FROM "LiteLLM_AccountingDelivery" WHERE state <> 'applied') THEN
            RAISE EXCEPTION 'Drain pending and parked accounting events before replacing delivery receipts';
        END IF;
        DROP TABLE IF EXISTS "LiteLLM_AccountingDelivery";
    END IF;
END $$;
DROP TYPE IF EXISTS "AccountingDeliveryTarget";
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
CREATE TABLE IF NOT EXISTS "LiteLLM_AccountingDelivery" (
    "request_id" TEXT PRIMARY KEY,
    "state" "AccountingDeliveryState" NOT NULL DEFAULT 'pending',
    "attempts" INTEGER NOT NULL DEFAULT 0,
    "next_attempt_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "applied_at" TIMESTAMP(3),
    "last_error" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "previous_row" JSONB
);
CREATE INDEX IF NOT EXISTS "LiteLLM_AccountingDelivery_state_next_attempt_at_idx"
    ON "LiteLLM_AccountingDelivery" ("state", "next_attempt_at");
CREATE INDEX IF NOT EXISTS "LiteLLM_AccountingDelivery_state_applied_at_idx"
    ON "LiteLLM_AccountingDelivery" ("state", "applied_at");
COMMIT;
