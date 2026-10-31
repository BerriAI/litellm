ALTER TABLE "LiteLLM_AccountingEvent"
ADD COLUMN IF NOT EXISTS "compilation_state" TEXT NOT NULL DEFAULT 'pending',
ADD COLUMN IF NOT EXISTS "compilation_error" TEXT,
ADD COLUMN IF NOT EXISTS "compilation_retry_at" TIMESTAMP(3);
CREATE INDEX IF NOT EXISTS "LiteLLM_AccountingEvent_kind_compilation_state_recorded_at_idx"
ON "LiteLLM_AccountingEvent"("kind", "compilation_state", "recorded_at");
