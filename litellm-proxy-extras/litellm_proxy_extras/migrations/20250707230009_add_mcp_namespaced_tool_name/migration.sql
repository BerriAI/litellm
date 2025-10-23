-- DropIndex
DROP INDEX "LiteLLM_DailyTagSpend_tag_date_api_key_model_custom_llm_pro_key";

-- DropIndex
DROP INDEX "LiteLLM_DailyTeamSpend_team_id_date_api_key_model_custom_ll_key";

-- DropIndex
DROP INDEX "LiteLLM_DailyUserSpend_user_id_date_api_key_model_custom_ll_key";

-- AlterTable
ALTER TABLE "LiteLLM_DailyTagSpend" ADD COLUMN     "mcp_namespaced_tool_name" TEXT,
ALTER COLUMN "model" DROP NOT NULL;

-- AlterTable
ALTER TABLE "LiteLLM_DailyTeamSpend" ADD COLUMN     "mcp_namespaced_tool_name" TEXT,
ALTER COLUMN "model" DROP NOT NULL;

-- AlterTable
ALTER TABLE "LiteLLM_DailyUserSpend" ADD COLUMN     "mcp_namespaced_tool_name" TEXT,
ALTER COLUMN "model" DROP NOT NULL;

-- AlterTable
ALTER TABLE "LiteLLM_SpendLogs" ADD COLUMN     "mcp_namespaced_tool_name" TEXT;

-- CreateIndex
CREATE INDEX "LiteLLM_DailyTagSpend_mcp_namespaced_tool_name_idx" ON "LiteLLM_DailyTagSpend"("mcp_namespaced_tool_name");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_DailyTagSpend_tag_date_api_key_model_custom_llm_pro_key" ON "LiteLLM_DailyTagSpend"("tag", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyTeamSpend_mcp_namespaced_tool_name_idx" ON "LiteLLM_DailyTeamSpend"("mcp_namespaced_tool_name");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_DailyTeamSpend_team_id_date_api_key_model_custom_ll_key" ON "LiteLLM_DailyTeamSpend"("team_id", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyUserSpend_mcp_namespaced_tool_name_idx" ON "LiteLLM_DailyUserSpend"("mcp_namespaced_tool_name");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_DailyUserSpend_user_id_date_api_key_model_custom_ll_key" ON "LiteLLM_DailyUserSpend"("user_id", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name");

