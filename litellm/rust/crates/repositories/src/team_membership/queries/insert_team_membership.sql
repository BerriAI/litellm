INSERT INTO "LiteLLM_TeamMembership" AS tm (user_id, team_id, budget_id)
VALUES ($1, $2, $3)
ON CONFLICT (user_id, team_id) DO UPDATE SET user_id = EXCLUDED.user_id
RETURNING tm.user_id,
          tm.team_id,
          tm.budget_id,
          tm.spend,
          tm.total_spend,
          (SELECT to_jsonb(b) FROM "LiteLLM_BudgetTable" AS b WHERE b.budget_id = tm.budget_id)
              AS "litellm_budget_table?: Json<Budget>"
