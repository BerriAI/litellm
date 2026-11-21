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
       NULL::jsonb AS "litellm_budget_table?: Json<Budget>"
FROM "LiteLLM_OrganizationTable" AS o
WHERE o.organization_alias = $1
ORDER BY o.organization_id
LIMIT 2
