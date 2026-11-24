SELECT access_group_id,
       access_group_name,
       description,
       COALESCE(access_model_names, '{}') AS "access_model_names!",
       COALESCE(access_mcp_server_ids, '{}') AS "access_mcp_server_ids!",
       mcp_tool_permissions AS "mcp_tool_permissions: RawJson",
       COALESCE(access_agent_ids, '{}') AS "access_agent_ids!",
       COALESCE(access_skill_names, '{}') AS "access_skill_names!",
       COALESCE(access_vector_store_ids, '{}') AS "access_vector_store_ids!",
       COALESCE(assigned_team_ids, '{}') AS "assigned_team_ids!",
       COALESCE(assigned_key_ids, '{}') AS "assigned_key_ids!",
       created_at,
       created_by,
       updated_at,
       updated_by
FROM "LiteLLM_AccessGroupTable"
WHERE access_group_id = $1
