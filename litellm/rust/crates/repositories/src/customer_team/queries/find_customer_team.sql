SELECT t.customer_team_id,
       t.alias,
       t.parent_org_id,
       t.spend,
       t.spend_reset_window,
       t.budget_id,
       t.budget_override_id,
       t.budget_override_expires_at,
       t.created_at,
       t.created_by,
       t.updated_at,
       t.updated_by,
       to_jsonb(b) AS "budget?: Json<Budget>",
       to_jsonb(o) AS "budget_override?: Json<Budget>"
FROM "LiteLLM_CustomerTeamTable" AS t
LEFT JOIN "LiteLLM_BudgetTable" AS b ON b.budget_id = t.budget_id
LEFT JOIN "LiteLLM_BudgetTable" AS o ON o.budget_id = t.budget_override_id
WHERE t.customer_team_id = $1
