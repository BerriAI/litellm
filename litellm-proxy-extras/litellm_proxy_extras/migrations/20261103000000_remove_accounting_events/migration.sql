BEGIN;
SET LOCAL lock_timeout = '10s';
DO $$
BEGIN
    IF to_regclass('"LiteLLM_AccountingEvent"') IS NOT NULL THEN
        LOCK TABLE "LiteLLM_AccountingEvent", "LiteLLM_AccountingDelivery" IN ACCESS EXCLUSIVE MODE;
        IF EXISTS (SELECT 1 FROM "LiteLLM_AccountingDelivery" WHERE state <> 'applied') THEN
            RAISE EXCEPTION 'Drain pending and parked accounting events before removing the event ledger';
        END IF;
        DROP TABLE "LiteLLM_AccountingEvent";
    END IF;
END $$;
DROP TYPE IF EXISTS "AccountingEventKind";
COMMIT;
