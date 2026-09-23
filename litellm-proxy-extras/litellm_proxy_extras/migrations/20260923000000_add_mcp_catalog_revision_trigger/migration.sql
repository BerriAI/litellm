CREATE OR REPLACE FUNCTION litellm_bump_mcp_catalog_revision() RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO "LiteLLM_Config" ("param_name", "reload_revision")
  VALUES ('mcp_catalog', 1)
  ON CONFLICT ("param_name") DO UPDATE SET "reload_revision" = "LiteLLM_Config"."reload_revision" + 1;
  RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS "litellm_mcp_catalog_revision" ON "LiteLLM_MCPServerTable";
CREATE TRIGGER "litellm_mcp_catalog_revision"
AFTER INSERT OR UPDATE OR DELETE ON "LiteLLM_MCPServerTable"
FOR EACH STATEMENT EXECUTE PROCEDURE litellm_bump_mcp_catalog_revision();
