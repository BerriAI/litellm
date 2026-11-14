CREATE INDEX CONCURRENTLY IF NOT EXISTS "LiteLLM_ManagedFileTable_created_by_created_at_idx"
ON "LiteLLM_ManagedFileTable" ("created_by", "created_at" DESC);
