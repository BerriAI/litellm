SELECT to_jsonb(a) || jsonb_build_object('extra_headers', COALESCE(to_jsonb(a.extra_headers), '[]'::jsonb),
    'agent_access_groups', COALESCE(to_jsonb(a.agent_access_groups), '[]'::jsonb),
    'access_group_ids', COALESCE(to_jsonb(a.access_group_ids), '[]'::jsonb), 'identity', (SELECT to_jsonb(i) || jsonb_build_object('required_roles', COALESCE(to_jsonb(i.required_roles), '[]'::jsonb),
    'required_scopes', COALESCE(to_jsonb(i.required_scopes), '[]'::jsonb)) FROM "LiteLLM_AgentIdentity" i WHERE i.agent_id=a.agent_id), 'object_permission', (SELECT to_jsonb(op) || jsonb_build_object('mcp_servers', COALESCE(to_jsonb(op.mcp_servers), '[]'::jsonb),
    'mcp_access_groups', COALESCE(to_jsonb(op.mcp_access_groups), '[]'::jsonb),
    'mcp_toolsets', COALESCE(to_jsonb(op.mcp_toolsets), '[]'::jsonb),
    'blocked_tools', COALESCE(to_jsonb(op.blocked_tools), '[]'::jsonb),
    'vector_stores', COALESCE(to_jsonb(op.vector_stores), '[]'::jsonb),
    'agents', COALESCE(to_jsonb(op.agents), '[]'::jsonb),
    'agent_access_groups', COALESCE(to_jsonb(op.agent_access_groups), '[]'::jsonb),
    'models', COALESCE(to_jsonb(op.models), '[]'::jsonb),
    'search_tools', COALESCE(to_jsonb(op.search_tools), '[]'::jsonb),
    'skills', COALESCE(to_jsonb(op.skills), '[]'::jsonb)) FROM "LiteLLM_ObjectPermissionTable" op WHERE op.object_permission_id = a.object_permission_id)) AS "data!: AuthData"
FROM "LiteLLM_AgentsTable" a
WHERE a.agent_id = ANY($1::text[])
ORDER BY a.agent_id
