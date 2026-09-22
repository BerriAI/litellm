-- CreateIndex
--
-- Built without CONCURRENTLY: db_scripts/partition_spend_logs.sql turns LiteLLM_SpendLogs into a
-- partitioned table, and PostgreSQL refuses CREATE INDEX CONCURRENTLY on a partitioned parent
-- (SQLSTATE 0A000). A plain build holds a SHARE lock on the table for the duration of the build.
CREATE INDEX IF NOT EXISTS "LiteLLM_SpendLogs_litellm_call_id_idx" ON "LiteLLM_SpendLogs"("litellm_call_id");
