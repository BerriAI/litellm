-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_AutoRouterSession" (
    "api_key" TEXT NOT NULL,
    "session_id" TEXT NOT NULL,
    "model_group" TEXT NOT NULL,
    "router_kind" TEXT NOT NULL,
    "baseline_model" TEXT,
    "first_turn_at" TIMESTAMP(3) NOT NULL,
    "last_turn_at" TIMESTAMP(3) NOT NULL,
    "turns" INTEGER NOT NULL DEFAULT 0,
    "turns_with_usage" INTEGER NOT NULL DEFAULT 0,
    "total_tokens" BIGINT NOT NULL DEFAULT 0,
    "ephemeral_5m_tokens" BIGINT NOT NULL DEFAULT 0,
    "ephemeral_1h_tokens" BIGINT NOT NULL DEFAULT 0,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "baseline_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "same_model_turns" INTEGER NOT NULL DEFAULT 0,
    "same_model_hits" INTEGER NOT NULL DEFAULT 0,
    "first_visit_turns" INTEGER NOT NULL DEFAULT 0,
    "first_visit_hits" INTEGER NOT NULL DEFAULT 0,
    "return_turns" INTEGER NOT NULL DEFAULT 0,
    "return_hits" INTEGER NOT NULL DEFAULT 0,
    "stale_return_misses" INTEGER NOT NULL DEFAULT 0,
    "savable_return_misses" INTEGER NOT NULL DEFAULT 0,
    "rescued_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "replay_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "last_model" TEXT,
    "model_state" JSONB NOT NULL DEFAULT '{}',
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_AutoRouterSession_pkey" PRIMARY KEY ("api_key","session_id","model_group")
);

-- CreateIndex
CREATE INDEX IF NOT EXISTS "idx_auto_router_session_group_activity" ON "LiteLLM_AutoRouterSession"("model_group", "last_turn_at");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "idx_auto_router_session_last_turn" ON "LiteLLM_AutoRouterSession"("last_turn_at");
