-- CreateIndex
-- Plain build on purpose: Postgres refuses CONCURRENTLY on a partitioned parent (what
-- db_scripts/partition_spend_logs.sql leaves behind) and inside the transaction prisma migrate
-- deploy runs this in. Inserts to "LiteLLM_SpendLogs" block for the build; operators can
-- pre-create the index under this exact name and this statement then no-ops.
CREATE INDEX IF NOT EXISTS "LiteLLM_SpendLogs_user_startTime_idx" ON "LiteLLM_SpendLogs"("user", "startTime" DESC);
