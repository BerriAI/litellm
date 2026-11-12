SET LOCAL lock_timeout = '10s';
CREATE TABLE IF NOT EXISTS "LiteLLM_DailyCustomerMemberSpend" (
    "id" TEXT NOT NULL,
    "customer_member_id" TEXT NOT NULL,
    "customer_team_id" TEXT NOT NULL,
    "customer_org_id" TEXT NOT NULL,
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
    "guardrail_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "api_requests" BIGINT NOT NULL DEFAULT 0,
    "successful_requests" BIGINT NOT NULL DEFAULT 0,
    "failed_requests" BIGINT NOT NULL DEFAULT 0,
    "total_response_time_ms" BIGINT NOT NULL DEFAULT 0,
    "timed_requests" BIGINT NOT NULL DEFAULT 0,
    "ptu_flat_cost" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_DailyCustomerMemberSpend_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "LiteLLM_DailyCustomerMemberSpend_date_idx" ON "LiteLLM_DailyCustomerMemberSpend"("date");

CREATE INDEX IF NOT EXISTS "LiteLLM_DailyCustomerMemberSpend_customer_member_id_date_idx" ON "LiteLLM_DailyCustomerMemberSpend"("customer_member_id", "date");

CREATE INDEX IF NOT EXISTS "LiteLLM_DailyCustomerMemberSpend_customer_team_id_date_idx" ON "LiteLLM_DailyCustomerMemberSpend"("customer_team_id", "date");

CREATE INDEX IF NOT EXISTS "LiteLLM_DailyCustomerMemberSpend_customer_org_id_date_idx" ON "LiteLLM_DailyCustomerMemberSpend"("customer_org_id", "date");

CREATE INDEX IF NOT EXISTS "LiteLLM_DailyCustomerMemberSpend_api_key_idx" ON "LiteLLM_DailyCustomerMemberSpend"("api_key");

CREATE INDEX IF NOT EXISTS "LiteLLM_DailyCustomerMemberSpend_model_idx" ON "LiteLLM_DailyCustomerMemberSpend"("model");

CREATE INDEX IF NOT EXISTS "LiteLLM_DailyCustomerMemberSpend_mcp_namespaced_tool_name_idx" ON "LiteLLM_DailyCustomerMemberSpend"("mcp_namespaced_tool_name");

CREATE INDEX IF NOT EXISTS "LiteLLM_DailyCustomerMemberSpend_endpoint_idx" ON "LiteLLM_DailyCustomerMemberSpend"("endpoint");

CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_DailyCustomerMemberSpend_identity_key" ON "LiteLLM_DailyCustomerMemberSpend"("customer_member_id", "customer_team_id", "customer_org_id", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");
