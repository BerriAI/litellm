SELECT object_permission_id,
       COALESCE(mcp_servers, '{}') AS "mcp_servers!",
       COALESCE(mcp_access_groups, '{}') AS "mcp_access_groups!",
       mcp_tool_permissions AS "mcp_tool_permissions: RawJson",
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
