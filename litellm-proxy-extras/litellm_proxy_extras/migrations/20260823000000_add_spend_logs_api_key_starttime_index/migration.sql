-- CreateIndex
-- Builds the covering (api_key, startTime, spend) index directly, so installs
-- that have not applied this migration yet pay for one index build on
-- LiteLLM_SpendLogs rather than two. Installs that already applied the earlier
-- (api_key, startTime) version of this file are moved over by
-- 20260918120000_spend_logs_api_key_index_cover_spend.
CREATE INDEX IF NOT EXISTS "LiteLLM_SpendLogs_api_key_startTime_spend_idx" ON "LiteLLM_SpendLogs"("api_key", "startTime", "spend");
