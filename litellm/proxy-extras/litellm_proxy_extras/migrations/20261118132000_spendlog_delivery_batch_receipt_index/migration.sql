CREATE INDEX CONCURRENTLY IF NOT EXISTS "LiteLLM_SpendLogDelivery_is_batch_idx"
ON "LiteLLM_SpendLogDelivery" ("is_batch") WHERE "is_batch";
