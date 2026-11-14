SET LOCAL lock_timeout = '10s';

-- Seed the MCP catalog revision row. litellm_bump_mcp_catalog_revision only creates it on
-- the first write to "LiteLLM_MCPServerTable"; until then get_mcp_catalog_revision returns
-- None and every MCP request reloads the whole server catalog from the database.
INSERT INTO "LiteLLM_Config" ("param_name", "reload_revision")
VALUES ('mcp_catalog', 1)
ON CONFLICT ("param_name") DO NOTHING;
