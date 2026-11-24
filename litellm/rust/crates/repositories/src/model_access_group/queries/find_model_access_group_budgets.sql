SELECT g.access_group_name AS "access_group_name!",
       g.spend AS "spend!",
       b.max_budget,
       b.budget_limits AS "budget_limits?: RawJson",
       b.budget_id AS "budget_id?"
FROM "LiteLLM_ModelAccessGroupBudgetTable" AS g
LEFT JOIN "LiteLLM_BudgetTable" AS b ON g.budget_id = b.budget_id
WHERE g.access_group_name = ANY($1)
