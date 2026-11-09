SELECT to_jsonb(u) || jsonb_build_object('teams', COALESCE(to_jsonb(u.teams), '[]'::jsonb),
    'models', COALESCE(to_jsonb(u.models), '[]'::jsonb),
    'allowed_cache_controls', COALESCE(to_jsonb(u.allowed_cache_controls), '[]'::jsonb),
    'policies', COALESCE(to_jsonb(u.policies), '[]'::jsonb), 'organization_memberships', COALESCE((SELECT jsonb_agg(to_jsonb(om)) FROM "LiteLLM_OrganizationMembership" om WHERE om.user_id = u.user_id), '[]'::jsonb)) AS "data!: AuthData" FROM "LiteLLM_UserTable" u WHERE LOWER(u.user_email) = LOWER($1) ORDER BY u.user_id LIMIT 1
