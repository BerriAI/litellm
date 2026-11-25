SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_ObjectPermissionTable"
    ADD COLUMN IF NOT EXISTS mcp_event_scope_inherited BOOLEAN NOT NULL DEFAULT FALSE;

CREATE OR REPLACE FUNCTION litellm_explicit_mcp_scope() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.mcp_event_scope_inherited := FALSE;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS litellm_explicit_mcp_scope ON "LiteLLM_ObjectPermissionTable";
CREATE TRIGGER litellm_explicit_mcp_scope BEFORE UPDATE OF
    mcp_servers, mcp_access_groups, mcp_tool_permissions, mcp_toolsets, blocked_tools
    ON "LiteLLM_ObjectPermissionTable" FOR EACH ROW
    WHEN (OLD.mcp_event_scope_inherited)
    EXECUTE PROCEDURE litellm_explicit_mcp_scope();
