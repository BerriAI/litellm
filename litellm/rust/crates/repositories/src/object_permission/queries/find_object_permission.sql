SELECT object_permission_id,
       CASE WHEN mcp_event_scope_inherited AND cardinality(mcp_servers) = 0 THEN NULL
            ELSE COALESCE(mcp_servers, '{}') END AS "mcp_servers?",
       COALESCE(mcp_access_groups, '{}') AS "mcp_access_groups!",
       mcp_tool_permissions AS "mcp_tool_permissions: RawJson",
       mcp_event_permissions AS "mcp_event_permissions: RawJson",
       COALESCE(vector_stores, '{}') AS "vector_stores!",
       COALESCE(agents, '{}') AS "agents!",
       COALESCE(agent_access_groups, '{}') AS "agent_access_groups!",
       COALESCE(models, '{}') AS "models!",
       COALESCE(blocked_tools, '{}') AS "blocked_tools!",
       COALESCE(mcp_toolsets, '{}') AS "mcp_toolsets!",
       COALESCE(search_tools, '{}') AS "search_tools!",
       mcp_tool_search_enabled,
       COALESCE(skills, '{}') AS "skills!"
FROM "LiteLLM_ObjectPermissionTable"
WHERE object_permission_id = $1
