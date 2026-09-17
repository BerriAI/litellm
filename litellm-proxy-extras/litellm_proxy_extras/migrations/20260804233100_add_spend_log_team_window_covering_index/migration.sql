-- Match the team budget-window aggregate with the same index-only access path.
-- This migration is separate from the api-key index because each concurrent
-- index build must be applied outside a transaction as a single statement.
CREATE INDEX CONCURRENTLY IF NOT EXISTS "LiteLLM_SpendLogs_team_id_startTime_idx"
ON "LiteLLM_SpendLogs" ("team_id", "startTime") INCLUDE ("spend");
