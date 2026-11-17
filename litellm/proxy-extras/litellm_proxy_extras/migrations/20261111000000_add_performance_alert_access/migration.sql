SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_PerformanceAlertState"
    ADD COLUMN IF NOT EXISTS "enabled" BOOLEAN NOT NULL DEFAULT true,
    ADD COLUMN IF NOT EXISTS "alerts_used" INTEGER;
