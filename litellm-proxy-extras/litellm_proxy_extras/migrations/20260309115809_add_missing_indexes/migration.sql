-- CreateIndex
CREATE INDEX "LiteLLM_VerificationToken_key_alias_idx" ON "LiteLLM_VerificationToken"("key_alias");

-- CreateIndex
CREATE INDEX "LiteLLM_SpendLogs_user_startTime_idx" ON "LiteLLM_SpendLogs"("user", "startTime");
