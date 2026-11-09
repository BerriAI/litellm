SELECT to_jsonb(a) || jsonb_build_object('access_model_names', COALESCE(to_jsonb(a.access_model_names), '[]'::jsonb),
    'access_mcp_server_ids', COALESCE(to_jsonb(a.access_mcp_server_ids), '[]'::jsonb),
    'access_agent_ids', COALESCE(to_jsonb(a.access_agent_ids), '[]'::jsonb),
    'access_skill_names', COALESCE(to_jsonb(a.access_skill_names), '[]'::jsonb),
    'assigned_team_ids', COALESCE(to_jsonb(a.assigned_team_ids), '[]'::jsonb),
    'assigned_key_ids', COALESCE(to_jsonb(a.assigned_key_ids), '[]'::jsonb)) AS "data!: AuthData"
FROM "LiteLLM_AccessGroupTable" a
WHERE a.access_group_id = ANY($1::text[])
ORDER BY a.access_group_id
