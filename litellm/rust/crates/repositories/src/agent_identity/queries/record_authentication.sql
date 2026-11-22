UPDATE "LiteLLM_AgentIdentity" AS i
SET last_authenticated_at = NOW() AT TIME ZONE 'UTC'
FROM "LiteLLM_AgentsTable" AS a
WHERE i.agent_id = $1
  AND i.revision = $2
  AND i.active
  AND a.agent_id = i.agent_id
  AND a.enabled
  AND a.identity_managed
