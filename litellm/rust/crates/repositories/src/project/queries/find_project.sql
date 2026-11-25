SELECT p.project_id,
       p.project_alias,
       p.description,
       p.team_id,
       p.budget_id,
       p.metadata AS "metadata: RawJson",
       COALESCE(p.models, '{}') AS "models!",
       p.spend,
       p.model_spend AS "model_spend: RawJson",
       p.model_rpm_limit AS "model_rpm_limit: RawJson",
       p.model_tpm_limit AS "model_tpm_limit: RawJson",
       p.blocked,
       p.object_permission_id,
       p.created_at,
       p.created_by,
       p.updated_at,
       p.updated_by,
       to_jsonb(b) AS "litellm_budget_table?: Json<Budget>"
FROM "LiteLLM_ProjectTable" AS p
LEFT JOIN "LiteLLM_BudgetTable" AS b ON b.budget_id = p.budget_id
WHERE p.project_id = $1
