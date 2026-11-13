SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_DailyCustomerMemberSpend" ALTER COLUMN "model_group" DROP NOT NULL;
ALTER TABLE "LiteLLM_DailyCustomerMemberSpend" ALTER COLUMN "model_group" DROP DEFAULT;

CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_DailyCustomerMemberSpend_identity_key"
    ON "LiteLLM_DailyCustomerMemberSpend"("customer_member_id", "customer_team_id", "customer_org_id", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");
