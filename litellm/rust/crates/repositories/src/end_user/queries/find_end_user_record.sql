SELECT e.user_id,
       e.alias,
       e.spend,
       e.allowed_model_region,
       e.default_model,
       COALESCE(e.models, '{}') AS "models!",
       e.budget_id,
       to_jsonb(b) AS "litellm_budget_table?: Json<Budget>",
       e.object_permission_id,
       to_jsonb(p) AS "object_permission?: Json<ObjectPermission>",
       e.blocked
FROM "LiteLLM_EndUserTable" AS e
LEFT JOIN "LiteLLM_BudgetTable" AS b ON b.budget_id = e.budget_id
LEFT JOIN "LiteLLM_ObjectPermissionTable" AS p ON p.object_permission_id = e.object_permission_id
WHERE e.user_id = $1
