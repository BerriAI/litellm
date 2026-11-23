INSERT INTO "LiteLLM_UserTable" AS u (
    user_id,
    user_email,
    sso_user_id,
    user_role,
    max_budget,
    budget_duration,
    budget_reset_at,
    models,
    budget_limits,
    tpm_limit,
    rpm_limit,
    max_parallel_requests,
    user_alias,
    team_id,
    organization_id,
    object_permission_id,
    password_reset_required,
    metadata,
    allowed_cache_controls,
    policies,
    model_max_budget,
    created_at,
    updated_at
)
VALUES (
    $1,
    CASE
        WHEN $3::bool OR NOT EXISTS (
            SELECT 1
            FROM "LiteLLM_UserTable"
            WHERE lower(user_email) = lower($2)
              AND user_id <> $1
        ) THEN $2
    END,
    $4,
    $5,
    $6,
    $7,
    $8,
    $9,
    $10,
    $11,
    $12,
    $13,
    $14,
    $15,
    $16,
    $17,
    $18,
    COALESCE($19::jsonb, '{}'),
    COALESCE($20::text[], '{}'),
    COALESCE($21::text[], '{}'),
    COALESCE($22::jsonb, '{}'),
    NOW() AT TIME ZONE 'UTC',
    NOW() AT TIME ZONE 'UTC'
)
ON CONFLICT DO NOTHING
RETURNING u.user_id,
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
       u.rollover_max_budget,
       u.rollover_credit,
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
