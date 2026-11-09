SELECT to_jsonb(i) || jsonb_build_object('required_roles', COALESCE(to_jsonb(i.required_roles), '[]'::jsonb),
    'required_scopes', COALESCE(to_jsonb(i.required_scopes), '[]'::jsonb)) AS "data!: AuthData" FROM "LiteLLM_AgentIdentity" i WHERE i.provider=$1 AND i.tenant_id=$2 AND i.client_id=$3
