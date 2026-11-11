ALTER TABLE "LiteLLM_PerformanceAlertState"
    ADD COLUMN "enabled" BOOLEAN NOT NULL DEFAULT true,
    ADD COLUMN "alerts_used" INTEGER;
