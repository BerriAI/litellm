ALTER TABLE "LiteLLM_AccountingEvent"
ADD COLUMN "compilation_state" TEXT NOT NULL DEFAULT 'pending',
ADD COLUMN "compilation_error" TEXT,
ADD COLUMN "compilation_retry_at" TIMESTAMP(3);
CREATE INDEX "LiteLLM_AccountingEvent_kind_compilation_state_recorded_at_idx"
ON "LiteLLM_AccountingEvent"("kind", "compilation_state", "recorded_at");
