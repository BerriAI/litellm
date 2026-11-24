SET LOCAL lock_timeout = '10s';

-- Keep the reload fallback if the database was baselined without the trigger.
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM pg_catalog.pg_trigger
    WHERE tgrelid = '"LiteLLM_MCPServerTable"'::regclass
      AND tgname = 'litellm_mcp_catalog_revision'
      AND tgenabled IN ('O', 'A')
  ) THEN
    DROP TRIGGER IF EXISTS "litellm_mcp_oauth_client_revision" ON "LiteLLM_MCPServerOAuthClient";
    CREATE TRIGGER "litellm_mcp_oauth_client_revision"
    AFTER INSERT OR UPDATE OR DELETE ON "LiteLLM_MCPServerOAuthClient"
    FOR EACH STATEMENT EXECUTE PROCEDURE litellm_bump_mcp_catalog_revision();

    INSERT INTO "LiteLLM_Config" ("param_name", "reload_revision")
    VALUES ('mcp_catalog', 1)
    ON CONFLICT ("param_name") DO NOTHING;
  END IF;
END;
$$ LANGUAGE plpgsql;
