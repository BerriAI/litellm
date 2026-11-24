SET LOCAL lock_timeout = '10s';

-- Keep the reload fallback if the database was baselined without the trigger.
-- data-migration-ok: The SELECT can insert only the single mcp_catalog key.
INSERT INTO "LiteLLM_Config" ("param_name", "reload_revision")
SELECT 'mcp_catalog', 1
WHERE EXISTS (
  SELECT 1 FROM pg_catalog.pg_trigger
  WHERE tgrelid = '"LiteLLM_MCPServerTable"'::regclass
    AND tgname = 'litellm_mcp_catalog_revision'
    AND tgenabled IN ('O', 'A')
)
ON CONFLICT ("param_name") DO NOTHING;
