SELECT to_jsonb(i) AS "data!: AuthData" FROM "LiteLLM_RetiredAgentIdentity" i WHERE i.provider=$1 AND i.tenant_id=$2 AND i.client_id=$3
