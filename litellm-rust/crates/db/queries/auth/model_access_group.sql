SELECT to_jsonb(g) || jsonb_build_object('litellm_budget_table', (SELECT to_jsonb(b) || jsonb_build_object('allowed_models', COALESCE(to_jsonb(b.allowed_models), '[]'::jsonb)) FROM "LiteLLM_BudgetTable" b WHERE b.budget_id = g.budget_id)) AS "data!: AuthData"
FROM "LiteLLM_ModelAccessGroupBudgetTable" g
WHERE g.access_group_name = ANY($1::text[])
ORDER BY g.access_group_name
