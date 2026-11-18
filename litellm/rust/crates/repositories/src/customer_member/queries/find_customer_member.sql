SELECT m.customer_member_id,
       m.alias,
       m.parent_org_id,
       COALESCE(m.parent_team_ids, '{}') AS "parent_team_ids!",
       m.spend,
       m.spend_reset_window,
       m.budget_id,
       m.budget_override_id,
       m.budget_override_expires_at,
       m.created_at,
       m.created_by,
       m.updated_at,
       m.updated_by,
       to_jsonb(b) AS "budget?: Json<Budget>",
       to_jsonb(o) AS "budget_override?: Json<Budget>"
FROM "LiteLLM_CustomerMemberTable" AS m
LEFT JOIN "LiteLLM_BudgetTable" AS b ON b.budget_id = m.budget_id
LEFT JOIN "LiteLLM_BudgetTable" AS o ON o.budget_id = m.budget_override_id
WHERE m.customer_member_id = $1
