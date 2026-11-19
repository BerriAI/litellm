SELECT u.user_id,
       u.user_alias,
       u.team_id,
       u.sso_user_id,
       u.organization_id,
       u.object_permission_id,
       u.password_reset_required,
       u.last_breach_check_at,
       COALESCE(u.teams, '{}') AS "teams!",
       u.user_role,
       u.max_budget,
       u.budget_limits AS "budget_limits: RawJson",
       u.spend,
       u.user_email,
       COALESCE(u.models, '{}') AS "models!",
       u.metadata AS "metadata: RawJson",
       u.max_parallel_requests,
       u.tpm_limit,
       u.rpm_limit,
       u.budget_duration,
       u.budget_reset_at,
       COALESCE(u.allowed_cache_controls, '{}') AS "allowed_cache_controls!",
       COALESCE(u.policies, '{}') AS "policies!",
       u.model_spend AS "model_spend: RawJson",
       u.model_max_budget AS "model_max_budget: RawJson",
       u.created_at,
       u.updated_at,
       COALESCE(
           (SELECT jsonb_agg(to_jsonb(m) ORDER BY m.organization_id)
            FROM "LiteLLM_OrganizationMembership" AS m
            WHERE m.user_id = u.user_id),
           '[]'
       ) AS "organization_memberships!: Json<Vec<OrganizationMembership>>"
FROM "LiteLLM_UserTable" AS u
WHERE lower(u.user_email) = lower($1)
  AND (NOT $2::bool OR u.sso_user_id IS NULL)
ORDER BY u.user_id
LIMIT 1
