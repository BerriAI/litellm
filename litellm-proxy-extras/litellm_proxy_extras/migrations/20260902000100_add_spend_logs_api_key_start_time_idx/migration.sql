-- CreateIndex
-- Same constraints as 20260902000000_add_spend_logs_user_start_time_idx: plain build so
-- partitioned parents can take it, inserts block for the duration, pre-creating under this
-- name makes it a no-op.
CREATE INDEX IF NOT EXISTS "LiteLLM_SpendLogs_api_key_startTime_idx" ON "LiteLLM_SpendLogs"("api_key", "startTime" DESC);
