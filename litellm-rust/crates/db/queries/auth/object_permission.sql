SELECT to_jsonb(op) || jsonb_build_object('mcp_servers', COALESCE(to_jsonb(op.mcp_servers), '[]'::jsonb),
    'mcp_access_groups', COALESCE(to_jsonb(op.mcp_access_groups), '[]'::jsonb),
    'mcp_toolsets', COALESCE(to_jsonb(op.mcp_toolsets), '[]'::jsonb),
    'blocked_tools', COALESCE(to_jsonb(op.blocked_tools), '[]'::jsonb),
    'vector_stores', COALESCE(to_jsonb(op.vector_stores), '[]'::jsonb),
    'agents', COALESCE(to_jsonb(op.agents), '[]'::jsonb),
    'agent_access_groups', COALESCE(to_jsonb(op.agent_access_groups), '[]'::jsonb),
    'models', COALESCE(to_jsonb(op.models), '[]'::jsonb),
    'search_tools', COALESCE(to_jsonb(op.search_tools), '[]'::jsonb),
    'skills', COALESCE(to_jsonb(op.skills), '[]'::jsonb)) AS "data!: AuthData"
FROM "LiteLLM_ObjectPermissionTable" op
WHERE op.object_permission_id = ANY($1::text[])
ORDER BY op.object_permission_id
