-- AlterTable
ALTER TABLE "LiteLLM_SpendLogs" ADD COLUMN     "agent_id" TEXT;

-- CreateTable
CREATE TABLE "LiteLLM_DailyAgentSpend" (
    "id" TEXT NOT NULL,
    "agent_id" TEXT,
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

    CONSTRAINT "LiteLLM_DailyAgentSpend_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "LiteLLM_DailyAgentSpend_date_idx" ON "LiteLLM_DailyAgentSpend"("date");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyAgentSpend_agent_id_idx" ON "LiteLLM_DailyAgentSpend"("agent_id");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyAgentSpend_api_key_idx" ON "LiteLLM_DailyAgentSpend"("api_key");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyAgentSpend_model_idx" ON "LiteLLM_DailyAgentSpend"("model");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyAgentSpend_mcp_namespaced_tool_name_idx" ON "LiteLLM_DailyAgentSpend"("mcp_namespaced_tool_name");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_DailyAgentSpend_agent_id_date_api_key_model_custom__key" ON "LiteLLM_DailyAgentSpend"("agent_id", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name");

