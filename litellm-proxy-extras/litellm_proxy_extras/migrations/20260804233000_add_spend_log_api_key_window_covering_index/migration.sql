-- Keep budget-window reseeds off the wide spend-log heap, including for keys
-- that own most rows. CONCURRENTLY avoids blocking inserts on production tables.
-- This migration must remain a single statement because CREATE INDEX
-- CONCURRENTLY cannot run inside a transaction.
CREATE INDEX CONCURRENTLY IF NOT EXISTS "LiteLLM_SpendLogs_api_key_startTime_idx"
ON "LiteLLM_SpendLogs" ("api_key", "startTime") INCLUDE ("spend");
