SELECT t.team_id,
       t.team_alias,
       t.organization_id,
       COALESCE(t.admins, '{}') AS "admins!",
       COALESCE(t.members, '{}') AS "members!",
       t.members_with_roles AS "members_with_roles: RawJson",
       t.metadata AS "metadata: RawJson",
       t.max_budget,
       t.soft_budget,
       t.spend,
       COALESCE(t.models, '{}') AS "models!",
       t.max_parallel_requests,
       t.tpm_limit,
       t.rpm_limit,
       t.tpd_limit,
       t.budget_duration,
       t.budget_reset_at,
       t.blocked,
       t.created_at,
       t.updated_at,
       t.model_spend AS "model_spend: RawJson",
       t.model_max_budget AS "model_max_budget: RawJson",
       t.router_settings AS "router_settings: RawJson",
       COALESCE(t.team_member_permissions, '{}') AS "team_member_permissions!",
       COALESCE(t.access_group_ids, '{}') AS "access_group_ids!",
       COALESCE(t.policies, '{}') AS "policies!",
       COALESCE(t.default_team_member_models, '{}') AS "default_team_member_models!",
       t.budget_limits AS "budget_limits: RawJson",
       t.model_id,
       t.allow_team_guardrail_config,
       t.object_permission_id,
       CASE WHEN m.id IS NOT NULL THEN jsonb_build_object(
           'id', m.id, 'model_aliases', m.aliases, 'created_by', m.created_by, 'updated_by', m.updated_by
       ) END AS "litellm_model_table?: RawJson"
FROM "LiteLLM_TeamTable" AS t
LEFT JOIN "LiteLLM_ModelTable" AS m ON m.id = t.model_id
WHERE t.team_alias = $1
ORDER BY t.team_id
LIMIT 2
