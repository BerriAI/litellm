ALTER TABLE "LiteLLM_ShadowEvalJob"
    ADD COLUMN "failure_counts" JSONB NOT NULL DEFAULT '{}',
    ADD COLUMN "failure_reason" TEXT;
