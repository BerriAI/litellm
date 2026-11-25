SELECT tm.user_id,
       tm.team_id,
       tm.budget_id,
       tm.spend,
       tm.rollover_credit,
       tm.total_spend,
       to_jsonb(b) AS "litellm_budget_table?: Json<Budget>"
FROM "LiteLLM_TeamMembership" AS tm
LEFT JOIN "LiteLLM_BudgetTable" AS b ON b.budget_id = tm.budget_id
WHERE tm.user_id = $1 AND tm.team_id = $2
