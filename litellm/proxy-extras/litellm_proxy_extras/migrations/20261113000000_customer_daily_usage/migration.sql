SET LOCAL lock_timeout = '10s';

CREATE TABLE IF NOT EXISTS "LiteLLM_DailyCustomerMemberSpend" (
    "id" TEXT NOT NULL,
    "date" TEXT NOT NULL,
    "customer_org_id" TEXT NOT NULL,
    "customer_team_id" TEXT NOT NULL DEFAULT '',
    "customer_member_id" TEXT NOT NULL DEFAULT '',
    "model" TEXT NOT NULL DEFAULT '',
    "model_group" TEXT NOT NULL DEFAULT '',
    "custom_llm_provider" TEXT NOT NULL DEFAULT '',
    "mcp_namespaced_tool_name" TEXT NOT NULL DEFAULT '',
    "endpoint" TEXT NOT NULL DEFAULT '',
    "api_key" TEXT NOT NULL DEFAULT '',
    "prompt_tokens" BIGINT NOT NULL DEFAULT 0,
    "completion_tokens" BIGINT NOT NULL DEFAULT 0,
    "cache_read_input_tokens" BIGINT NOT NULL DEFAULT 0,
    "cache_creation_input_tokens" BIGINT NOT NULL DEFAULT 0,
    "compression_saved_tokens" BIGINT NOT NULL DEFAULT 0,
    "api_requests" BIGINT NOT NULL DEFAULT 0,
    "successful_requests" BIGINT NOT NULL DEFAULT 0,
    "failed_requests" BIGINT NOT NULL DEFAULT 0,
    "total_response_time_ms" BIGINT NOT NULL DEFAULT 0,
    "timed_requests" BIGINT NOT NULL DEFAULT 0,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "compression_savings_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "prompt_caching_savings_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "gateway_injected_caching_savings_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "autorouter_savings_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "guardrail_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "LiteLLM_DailyCustomerMemberSpend_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "LiteLLM_DailyCustomerMemberSpend_date_idx"
    ON "LiteLLM_DailyCustomerMemberSpend"("date");
CREATE INDEX IF NOT EXISTS "LiteLLM_DailyCustomerMemberSpend_customer_org_id_date_idx"
    ON "LiteLLM_DailyCustomerMemberSpend"("customer_org_id", "date");
CREATE INDEX IF NOT EXISTS "LiteLLM_DailyCustomerMemberSpend_customer_team_id_date_idx"
    ON "LiteLLM_DailyCustomerMemberSpend"("customer_team_id", "date");
CREATE INDEX IF NOT EXISTS "LiteLLM_DailyCustomerMemberSpend_customer_member_id_date_idx"
    ON "LiteLLM_DailyCustomerMemberSpend"("customer_member_id", "date");
CREATE INDEX IF NOT EXISTS "LiteLLM_DailyCustomerMemberSpend_api_key_idx"
    ON "LiteLLM_DailyCustomerMemberSpend"("api_key");
CREATE INDEX IF NOT EXISTS "LiteLLM_DailyCustomerMemberSpend_model_idx"
    ON "LiteLLM_DailyCustomerMemberSpend"("model");
