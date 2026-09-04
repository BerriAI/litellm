DROP INDEX IF EXISTS "LiteLLM_DailyAgentSpend_agent_id_date_api_key_model_custom__key";

DROP INDEX IF EXISTS "LiteLLM_DailyEndUserSpend_end_user_id_date_api_key_model_cu_key";

DROP INDEX IF EXISTS "LiteLLM_DailyOrganizationSpend_organization_id_date_api_key_key";

DROP INDEX IF EXISTS "LiteLLM_DailyTagSpend_tag_date_api_key_model_custom_llm_pro_key";

DROP INDEX IF EXISTS "LiteLLM_DailyTeamSpend_team_id_date_api_key_model_custom_ll_key";

DROP INDEX IF EXISTS "LiteLLM_DailyUserSpend_user_id_date_api_key_model_custom_ll_key";

UPDATE "LiteLLM_DailyAgentSpend" SET "model_group" = '' WHERE "model_group" IS NULL;

UPDATE "LiteLLM_DailyEndUserSpend" SET "model_group" = '' WHERE "model_group" IS NULL;

UPDATE "LiteLLM_DailyOrganizationSpend" SET "model_group" = '' WHERE "model_group" IS NULL;

UPDATE "LiteLLM_DailyTagSpend" SET "model_group" = '' WHERE "model_group" IS NULL;

UPDATE "LiteLLM_DailyTeamSpend" SET "model_group" = '' WHERE "model_group" IS NULL;

UPDATE "LiteLLM_DailyUserSpend" SET "model_group" = '' WHERE "model_group" IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_DailyAgentSpend_agent_id_date_api_key_model_custom__key" ON "LiteLLM_DailyAgentSpend"("agent_id", "date", "api_key", "model", "model_group", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");

CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_DailyEndUserSpend_end_user_id_date_api_key_model_cu_key" ON "LiteLLM_DailyEndUserSpend"("end_user_id", "date", "api_key", "model", "model_group", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");

CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_DailyOrganizationSpend_organization_id_date_api_key_key" ON "LiteLLM_DailyOrganizationSpend"("organization_id", "date", "api_key", "model", "model_group", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");

CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_DailyTagSpend_tag_date_api_key_model_custom_llm_pro_key" ON "LiteLLM_DailyTagSpend"("tag", "date", "api_key", "model", "model_group", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");

CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_DailyTeamSpend_team_id_date_api_key_model_custom_ll_key" ON "LiteLLM_DailyTeamSpend"("team_id", "date", "api_key", "model", "model_group", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");

CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_DailyUserSpend_user_id_date_api_key_model_custom_ll_key" ON "LiteLLM_DailyUserSpend"("user_id", "date", "api_key", "model", "model_group", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");
