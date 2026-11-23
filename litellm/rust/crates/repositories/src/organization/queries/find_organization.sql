SELECT o.organization_id,
       o.organization_alias,
       o.budget_id,
       o.metadata AS "metadata: RawJson",
       COALESCE(o.models, '{}') AS "models!",
       o.spend,
       o.model_spend AS "model_spend: RawJson",
       o.object_permission_id,
       o.created_by,
       o.updated_by,
       to_jsonb(b) AS "litellm_budget_table?: Json<Budget>"
FROM "LiteLLM_OrganizationTable" AS o
LEFT JOIN "LiteLLM_BudgetTable" AS b ON b.budget_id = o.budget_id AND $2
WHERE o.organization_id = $1
