SELECT to_jsonb(tm) || jsonb_build_object('litellm_budget_table', (SELECT to_jsonb(b) || jsonb_build_object('allowed_models', COALESCE(to_jsonb(b.allowed_models), '[]'::jsonb)) FROM "LiteLLM_BudgetTable" b WHERE b.budget_id = tm.budget_id)) AS "data!: AuthData"
FROM "LiteLLM_TeamMembership" tm WHERE tm.user_id=$1 AND tm.team_id=$2
