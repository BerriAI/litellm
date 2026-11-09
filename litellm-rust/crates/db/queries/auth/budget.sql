SELECT to_jsonb(b) || jsonb_build_object('allowed_models', COALESCE(to_jsonb(b.allowed_models), '[]'::jsonb)) AS "data!: AuthData"
FROM "LiteLLM_BudgetTable" b
WHERE b.budget_id = ANY($1::text[])
ORDER BY b.budget_id
