SELECT v.token,
       v.key_name,
       v.key_alias,
       v.soft_budget_cooldown,
       v.spend,
       v.total_spend,
       v.expires,
       COALESCE(v.models, '{}') AS "models!",
       v.aliases AS "aliases: RawJson",
       v.config AS "config: RawJson",
       v.router_settings AS "router_settings: RawJson",
       v.user_id,
       v.team_id,
       v.agent_id,
       v.project_id,
       v.permissions AS "permissions: RawJson",
       v.max_parallel_requests,
       v.metadata AS "metadata: RawJson",
       v.blocked,
       COALESCE(v.tpm_limit, b.tpm_limit) AS tpm_limit,
       COALESCE(v.rpm_limit, b.rpm_limit) AS rpm_limit,
       COALESCE(v.tpd_limit, b.tpd_limit) AS tpd_limit,
       COALESCE(v.max_budget, b.max_budget) AS max_budget,
       b.soft_budget AS soft_budget,
       v.budget_duration,
       v.budget_reset_at,
       v.allowed_cache_controls,
       v.allowed_routes,
       v.key_type,
       v.access_group_ids,
       v.model_spend AS "model_spend: RawJson",
       CASE WHEN v.model_max_budget = '{}'::jsonb
            THEN COALESCE(b.model_max_budget, v.model_max_budget)
            ELSE v.model_max_budget END AS "model_max_budget!: RawJson",
       v.budget_fallbacks AS "budget_fallbacks: RawJson",
       v.budget_id,
       COALESCE(v.organization_id, t.organization_id) AS org_id,
       v.object_permission_id,
       v.created_at,
       v.created_by,
       v.updated_at,
       v.updated_by,
       v.settings_updated_at,
       v.last_active,
       v.rotation_count,
       v.auto_rotate,
       v.rotation_interval,
       v.last_rotation_at,
       v.key_rotation_at,
       v.budget_limits AS "budget_limits: RawJson",
       t.spend AS "team_spend?",
       t.max_budget AS team_max_budget,
       t.soft_budget AS team_soft_budget,
       t.model_max_budget AS "team_model_max_budget?: RawJson",
       t.tpm_limit AS team_tpm_limit,
       t.rpm_limit AS team_rpm_limit,
       t.tpd_limit AS team_tpd_limit,
       COALESCE(t.models, '{}') AS "team_models!",
       t.metadata AS "team_metadata?: RawJson",
       COALESCE(t.blocked, false) AS "team_blocked!",
       t.team_alias,
       (SELECT member.entry
          FROM jsonb_array_elements(CASE WHEN jsonb_typeof(t.members_with_roles) = 'array'
                                         THEN t.members_with_roles ELSE '[]'::jsonb END)
               WITH ORDINALITY AS member(entry, position)
         WHERE member.entry -> 'user_id' = to_jsonb(v.user_id)
         ORDER BY member.position DESC
         LIMIT 1) AS "team_member?: RawJson",
       t.object_permission_id AS team_object_permission_id,
       p.project_alias AS "project_alias?",
       tm.spend AS "team_member_spend?",
       b_tm.tpm_limit AS team_member_tpm_limit,
       b_tm.rpm_limit AS team_member_rpm_limit,
       m.aliases AS "team_model_aliases?: RawJson",
       o.metadata AS "organization_metadata?: RawJson",
       o.organization_alias AS "organization_alias?",
       b2.max_budget AS organization_max_budget,
       b2.tpm_limit AS organization_tpm_limit,
       b2.rpm_limit AS organization_rpm_limit
FROM "LiteLLM_VerificationToken" AS v
LEFT JOIN "LiteLLM_TeamTable" AS t ON v.team_id = t.team_id
LEFT JOIN "LiteLLM_TeamMembership" AS tm ON v.team_id = tm.team_id AND tm.user_id = v.user_id
LEFT JOIN "LiteLLM_BudgetTable" AS b_tm ON tm.budget_id = b_tm.budget_id
LEFT JOIN "LiteLLM_ModelTable" AS m ON t.model_id = m.id
LEFT JOIN "LiteLLM_BudgetTable" AS b ON v.budget_id = b.budget_id
LEFT JOIN "LiteLLM_ProjectTable" AS p ON v.project_id = p.project_id
LEFT JOIN "LiteLLM_OrganizationTable" AS o ON v.organization_id = o.organization_id
LEFT JOIN "LiteLLM_BudgetTable" AS b2 ON o.budget_id = b2.budget_id
WHERE v.token = $1
