SELECT budget_id AS id, budget_limits::jsonb AS "limits!: RawJson"
FROM "LiteLLM_BudgetTable"
WHERE budget_id > $1 AND jsonb_typeof(budget_limits::jsonb) = 'array'
ORDER BY budget_id
LIMIT $2
