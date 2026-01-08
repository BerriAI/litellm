-- DropIndex
DROP INDEX "LiteLLM_DailyAgentSpend_agent_id_date_api_key_model_custom__key";

-- DropIndex
DROP INDEX "LiteLLM_DailyEndUserSpend_end_user_id_date_api_key_model_cu_key";

-- DropIndex
DROP INDEX "LiteLLM_DailyOrganizationSpend_organization_id_date_api_key_key";

-- DropIndex
DROP INDEX "LiteLLM_DailyTagSpend_tag_date_api_key_model_custom_llm_pro_key";

-- DropIndex
DROP INDEX "LiteLLM_DailyTeamSpend_team_id_date_api_key_model_custom_ll_key";

-- DropIndex
DROP INDEX "LiteLLM_DailyUserSpend_user_id_date_api_key_model_custom_ll_key";

-- AlterTable
ALTER TABLE "LiteLLM_DailyAgentSpend" ADD COLUMN     "endpoint" TEXT;

-- AlterTable
ALTER TABLE "LiteLLM_DailyEndUserSpend" ADD COLUMN     "endpoint" TEXT;

-- AlterTable
ALTER TABLE "LiteLLM_DailyOrganizationSpend" ADD COLUMN     "endpoint" TEXT;

-- AlterTable
ALTER TABLE "LiteLLM_DailyTagSpend" ADD COLUMN     "endpoint" TEXT;

-- AlterTable
ALTER TABLE "LiteLLM_DailyTeamSpend" ADD COLUMN     "endpoint" TEXT;

-- AlterTable
ALTER TABLE "LiteLLM_DailyUserSpend" ADD COLUMN     "endpoint" TEXT;

-- CreateIndex
CREATE INDEX "LiteLLM_DailyAgentSpend_endpoint_idx" ON "LiteLLM_DailyAgentSpend"("endpoint");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_DailyAgentSpend_agent_id_date_api_key_model_custom__key" ON "LiteLLM_DailyAgentSpend"("agent_id", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyEndUserSpend_endpoint_idx" ON "LiteLLM_DailyEndUserSpend"("endpoint");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_DailyEndUserSpend_end_user_id_date_api_key_model_cu_key" ON "LiteLLM_DailyEndUserSpend"("end_user_id", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyOrganizationSpend_endpoint_idx" ON "LiteLLM_DailyOrganizationSpend"("endpoint");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_DailyOrganizationSpend_organization_id_date_api_key_key" ON "LiteLLM_DailyOrganizationSpend"("organization_id", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyTagSpend_endpoint_idx" ON "LiteLLM_DailyTagSpend"("endpoint");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_DailyTagSpend_tag_date_api_key_model_custom_llm_pro_key" ON "LiteLLM_DailyTagSpend"("tag", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyTeamSpend_endpoint_idx" ON "LiteLLM_DailyTeamSpend"("endpoint");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_DailyTeamSpend_team_id_date_api_key_model_custom_ll_key" ON "LiteLLM_DailyTeamSpend"("team_id", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyUserSpend_endpoint_idx" ON "LiteLLM_DailyUserSpend"("endpoint");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_DailyUserSpend_user_id_date_api_key_model_custom_ll_key" ON "LiteLLM_DailyUserSpend"("user_id", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");

