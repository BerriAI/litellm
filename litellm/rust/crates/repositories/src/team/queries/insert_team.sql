INSERT INTO "LiteLLM_TeamTable" (
    team_id,
    organization_id,
    admins,
    members,
    members_with_roles,
    models,
    blocked,
    allow_team_guardrail_config,
    model_spend,
    router_settings,
    max_budget,
    budget_duration,
    budget_reset_at,
    tpm_limit,
    rpm_limit,
    team_member_permissions,
    access_group_ids,
    created_at,
    updated_at
)
VALUES (
    $1,
    $2,
    '{}',
    '{}',
    '[]',
    '{}',
    false,
    false,
    '{}',
    '{}',
    $3,
    $4,
    $5,
    $6,
    $7,
    COALESCE($8::text[], '{}'),
    COALESCE($9::text[], '{}'),
    NOW() AT TIME ZONE 'UTC',
    NOW() AT TIME ZONE 'UTC'
)
ON CONFLICT DO NOTHING
RETURNING team_id
