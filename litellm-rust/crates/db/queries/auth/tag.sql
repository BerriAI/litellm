SELECT to_jsonb(t) || jsonb_build_object('models', COALESCE(to_jsonb(t.models), '[]'::jsonb), 'litellm_budget_table', (SELECT to_jsonb(b) || jsonb_build_object('allowed_models', COALESCE(to_jsonb(b.allowed_models), '[]'::jsonb)) FROM "LiteLLM_BudgetTable" b WHERE b.budget_id = t.budget_id)) AS "data!: AuthData"
FROM "LiteLLM_TagTable" t
WHERE t.tag_name = ANY($1::text[])
ORDER BY t.tag_name
