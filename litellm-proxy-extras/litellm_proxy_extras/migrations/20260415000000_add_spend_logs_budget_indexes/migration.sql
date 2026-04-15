-- CreateIndex: support fast per-key and per-end-user spend lookups
-- used by the cold-cache DB fallback in _PROXY_VirtualKeyModelMaxBudgetLimiter
CREATE INDEX IF NOT EXISTS "LiteLLM_SpendLogs_api_key_startTime_idx" ON "LiteLLM_SpendLogs"("api_key", "startTime");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_SpendLogs_end_user_startTime_idx" ON "LiteLLM_SpendLogs"("end_user", "startTime");
