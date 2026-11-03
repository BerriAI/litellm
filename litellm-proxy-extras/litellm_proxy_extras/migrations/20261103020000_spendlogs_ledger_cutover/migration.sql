BEGIN;
SET LOCAL lock_timeout = '5s';
ALTER TABLE "LiteLLM_AccountingDelivery" ADD COLUMN IF NOT EXISTS "legacy_takeover" BOOLEAN NOT NULL DEFAULT false;
COMMIT;
