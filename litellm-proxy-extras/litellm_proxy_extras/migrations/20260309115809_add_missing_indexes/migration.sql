-- SkipTransactionBlock

-- Drop invalid indexes left behind by failed CONCURRENTLY builds
DROP INDEX CONCURRENTLY IF EXISTS "LiteLLM_VerificationToken_key_alias_idx";

-- CreateIndex
CREATE INDEX CONCURRENTLY "LiteLLM_VerificationToken_key_alias_idx" ON "LiteLLM_VerificationToken"("key_alias");

-- Drop invalid indexes left behind by failed CONCURRENTLY builds
DROP INDEX CONCURRENTLY IF EXISTS "LiteLLM_SpendLogs_user_startTime_idx";

-- CreateIndex
CREATE INDEX CONCURRENTLY "LiteLLM_SpendLogs_user_startTime_idx" ON "LiteLLM_SpendLogs"("user", "startTime");
