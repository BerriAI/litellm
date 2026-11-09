-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_DailyTeamUserSpend" (
    "id" TEXT NOT NULL,
    "team_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "date" TEXT NOT NULL,
    "api_key" TEXT NOT NULL,
    "model" TEXT,
    "model_group" TEXT,
    "custom_llm_provider" TEXT,
    "mcp_namespaced_tool_name" TEXT,
    "endpoint" TEXT,
    "prompt_tokens" BIGINT NOT NULL DEFAULT 0,
    "completion_tokens" BIGINT NOT NULL DEFAULT 0,
    "cache_read_input_tokens" BIGINT NOT NULL DEFAULT 0,
    "cache_creation_input_tokens" BIGINT NOT NULL DEFAULT 0,
    "compression_saved_tokens" BIGINT NOT NULL DEFAULT 0,
    "compression_savings_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "prompt_caching_savings_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "gateway_injected_caching_savings_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "autorouter_savings_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "api_requests" BIGINT NOT NULL DEFAULT 0,
    "successful_requests" BIGINT NOT NULL DEFAULT 0,
    "failed_requests" BIGINT NOT NULL DEFAULT 0,
    "total_response_time_ms" BIGINT NOT NULL DEFAULT 0,
    "timed_requests" BIGINT NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_DailyTeamUserSpend_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_DailyTeamUserSpend_team_id_date_idx" ON "LiteLLM_DailyTeamUserSpend"("team_id", "date");

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_DailyTeamUserSpend_team_id_user_id_date_api_key_mod_key" ON "LiteLLM_DailyTeamUserSpend"("team_id", "user_id", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");

