-- prisma migrate deploy applies this file inside a transaction block, so
-- standard index DDL is required here.
DROP INDEX IF EXISTS "LiteLLM_VerificationToken_key_alias_idx";

-- CreateIndex
CREATE INDEX "LiteLLM_VerificationToken_key_alias_idx" ON "LiteLLM_VerificationToken"("key_alias");

DROP INDEX IF EXISTS "LiteLLM_SpendLogs_user_startTime_idx";

-- CreateIndex
CREATE INDEX "LiteLLM_SpendLogs_user_startTime_idx" ON "LiteLLM_SpendLogs"("user", "startTime");
