BEGIN;
SET LOCAL lock_timeout = '10s';
LOCK TABLE "LiteLLM_AccountingEvent", "LiteLLM_AccountingDelivery" IN ACCESS EXCLUSIVE MODE;
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM "LiteLLM_AccountingDelivery" WHERE state <> 'applied') THEN
        RAISE EXCEPTION 'Drain pending and parked accounting events before removing the event ledger';
    END IF;
END $$;
DROP TABLE "LiteLLM_AccountingEvent";
DROP TYPE "AccountingEventKind";
COMMIT;
