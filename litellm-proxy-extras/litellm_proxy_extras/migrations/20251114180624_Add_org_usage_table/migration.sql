-- CreateTable
CREATE TABLE "LiteLLM_DailyOrganizationSpend" (
    "id" TEXT NOT NULL,
    "organization_id" TEXT,
    "date" TEXT NOT NULL,
    "api_key" TEXT NOT NULL,
    "model" TEXT,
    "model_group" TEXT,
    "custom_llm_provider" TEXT,
    "mcp_namespaced_tool_name" TEXT,
    "prompt_tokens" BIGINT NOT NULL DEFAULT 0,
    "completion_tokens" BIGINT NOT NULL DEFAULT 0,
    "cache_read_input_tokens" BIGINT NOT NULL DEFAULT 0,
    "cache_creation_input_tokens" BIGINT NOT NULL DEFAULT 0,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "api_requests" BIGINT NOT NULL DEFAULT 0,
    "successful_requests" BIGINT NOT NULL DEFAULT 0,
    "failed_requests" BIGINT NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_DailyOrganizationSpend_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "LiteLLM_DailyOrganizationSpend_date_idx" ON "LiteLLM_DailyOrganizationSpend"("date");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyOrganizationSpend_organization_id_idx" ON "LiteLLM_DailyOrganizationSpend"("organization_id");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyOrganizationSpend_api_key_idx" ON "LiteLLM_DailyOrganizationSpend"("api_key");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyOrganizationSpend_model_idx" ON "LiteLLM_DailyOrganizationSpend"("model");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyOrganizationSpend_mcp_namespaced_tool_name_idx" ON "LiteLLM_DailyOrganizationSpend"("mcp_namespaced_tool_name");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_DailyOrganizationSpend_organization_id_date_api_key_key" ON "LiteLLM_DailyOrganizationSpend"("organization_id", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name");

