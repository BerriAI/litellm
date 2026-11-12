CREATE INDEX CONCURRENTLY IF NOT EXISTS "idx_background_interaction_settlement_settled_at"
ON "LiteLLM_BackgroundInteractionSettlement" ("settled_at");
