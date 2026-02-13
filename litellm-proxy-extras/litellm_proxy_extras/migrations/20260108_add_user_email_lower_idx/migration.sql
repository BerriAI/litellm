-- CreateIndex
-- Fixes performance issue in _check_duplicate_user_email function
-- by enabling fast case-insensitive email lookups.
-- 
-- Without this index, queries with mode: "insensitive" cause full table scans.
-- With this index, PostgreSQL can use an Index Scan for O(log n) performance.
--
-- Related: GitHub Issue #18411
CREATE INDEX "LiteLLM_UserTable_user_email_lower_idx" ON "LiteLLM_UserTable"(LOWER("user_email"));
