CREATE TABLE IF NOT EXISTS "LiteLLM_AutoRouterSession" (
    "api_key" TEXT NOT NULL,
    "session_id" TEXT NOT NULL,
    "router_name" TEXT NOT NULL,
    "router_type" TEXT NOT NULL,
    "first_turn_at" TIMESTAMP(3) NOT NULL,
    "last_turn_at" TIMESTAMP(3) NOT NULL,
    "last_model" TEXT NOT NULL,
    "models" JSONB NOT NULL DEFAULT '{}',
    "turns" INTEGER NOT NULL DEFAULT 0,
    "unordered_turns" INTEGER NOT NULL DEFAULT 0,
    "covered_turns" INTEGER NOT NULL DEFAULT 0,
    "cache_hits" INTEGER NOT NULL DEFAULT 0,
    "same_model_turns" INTEGER NOT NULL DEFAULT 0,
    "same_model_hits" INTEGER NOT NULL DEFAULT 0,
    "first_visit_turns" INTEGER NOT NULL DEFAULT 0,
    "first_visit_hits" INTEGER NOT NULL DEFAULT 0,
    "return_turns" INTEGER NOT NULL DEFAULT 0,
    "return_hits" INTEGER NOT NULL DEFAULT 0,
    "return_expired_misses" INTEGER NOT NULL DEFAULT 0,
    "return_within_ttl_misses" INTEGER NOT NULL DEFAULT 0,
    "ttl_5m_turns" INTEGER NOT NULL DEFAULT 0,
    "ttl_1h_turns" INTEGER NOT NULL DEFAULT 0,
    "total_tokens" BIGINT NOT NULL DEFAULT 0,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "saved_spend" DOUBLE PRECISION NOT NULL DEFAULT 0,

    CONSTRAINT "LiteLLM_AutoRouterSession_pkey" PRIMARY KEY ("api_key", "session_id", "router_name")
);

CREATE INDEX IF NOT EXISTS "idx_autorouter_session_last_turn" ON "LiteLLM_AutoRouterSession"("last_turn_at");
