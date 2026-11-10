CREATE INDEX CONCURRENTLY IF NOT EXISTS "LiteLLM_DailyUserSpend_api_key_date_idx"
ON "LiteLLM_DailyUserSpend" ("api_key", "date");
