SELECT to_jsonb(t) || jsonb_build_object('admins', COALESCE(to_jsonb(t.admins), '[]'::jsonb),
    'members', COALESCE(to_jsonb(t.members), '[]'::jsonb),
    'models', COALESCE(to_jsonb(t.models), '[]'::jsonb),
    'team_member_permissions', COALESCE(to_jsonb(t.team_member_permissions), '[]'::jsonb),
    'access_group_ids', COALESCE(to_jsonb(t.access_group_ids), '[]'::jsonb),
    'policies', COALESCE(to_jsonb(t.policies), '[]'::jsonb),
    'default_team_member_models', COALESCE(to_jsonb(t.default_team_member_models), '[]'::jsonb), 'litellm_model_table', (SELECT (to_jsonb(m) - 'aliases') || jsonb_build_object('model_aliases', m.aliases) FROM "LiteLLM_ModelTable" m WHERE m.id=t.model_id), 'object_permission', (SELECT to_jsonb(op) || jsonb_build_object('mcp_servers', COALESCE(to_jsonb(op.mcp_servers), '[]'::jsonb),
    'mcp_access_groups', COALESCE(to_jsonb(op.mcp_access_groups), '[]'::jsonb),
    'mcp_toolsets', COALESCE(to_jsonb(op.mcp_toolsets), '[]'::jsonb),
    'blocked_tools', COALESCE(to_jsonb(op.blocked_tools), '[]'::jsonb),
    'vector_stores', COALESCE(to_jsonb(op.vector_stores), '[]'::jsonb),
    'agents', COALESCE(to_jsonb(op.agents), '[]'::jsonb),
    'agent_access_groups', COALESCE(to_jsonb(op.agent_access_groups), '[]'::jsonb),
    'models', COALESCE(to_jsonb(op.models), '[]'::jsonb),
    'search_tools', COALESCE(to_jsonb(op.search_tools), '[]'::jsonb),
    'skills', COALESCE(to_jsonb(op.skills), '[]'::jsonb)) FROM "LiteLLM_ObjectPermissionTable" op WHERE op.object_permission_id = t.object_permission_id)) AS "data!: AuthData"
FROM "LiteLLM_TeamTable" t
WHERE t.team_id = ANY($1::text[])
ORDER BY t.team_id
