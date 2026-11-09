SELECT jsonb_build_object('agent_id', a.agent_id) AS "data!: AuthData"
FROM "LiteLLM_AgentsTable" a
WHERE a.agent_access_groups && $1::text[]
ORDER BY a.agent_id
