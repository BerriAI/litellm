BEGIN;
SET LOCAL lock_timeout = '10s';
LOCK TABLE "LiteLLM_AccountingDelivery" IN ACCESS EXCLUSIVE MODE;
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM "LiteLLM_AccountingDelivery" WHERE state <> 'applied') THEN
        RAISE EXCEPTION 'Drain pending and parked accounting events before replacing delivery receipts';
    END IF;
END $$;
DROP TABLE "LiteLLM_AccountingDelivery";
DROP TYPE "AccountingDeliveryTarget";
ALTER TABLE "LiteLLM_SpendLogs" ADD COLUMN "accounting_path" TEXT;
ALTER TABLE "LiteLLM_SpendLogs" ADD CONSTRAINT "LiteLLM_SpendLogs_accounting_path_check"
    CHECK (accounting_path IS NULL OR accounting_path IN ('legacy', 'ledger'));
CREATE TABLE "LiteLLM_AccountingDelivery" (
    "request_id" TEXT PRIMARY KEY,
    "state" "AccountingDeliveryState" NOT NULL DEFAULT 'pending',
    "attempts" INTEGER NOT NULL DEFAULT 0,
    "next_attempt_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "applied_at" TIMESTAMP(3),
    "last_error" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "previous_row" JSONB
);
CREATE INDEX "LiteLLM_AccountingDelivery_state_next_attempt_at_idx"
    ON "LiteLLM_AccountingDelivery" ("state", "next_attempt_at");
CREATE INDEX "LiteLLM_AccountingDelivery_state_applied_at_idx"
    ON "LiteLLM_AccountingDelivery" ("state", "applied_at");
COMMIT;
