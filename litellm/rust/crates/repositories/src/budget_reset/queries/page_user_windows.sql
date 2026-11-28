SELECT user_id AS id, budget_limits::jsonb AS "limits!: RawJson"
FROM "LiteLLM_UserTable"
WHERE user_id > $1 AND jsonb_typeof(budget_limits::jsonb) = 'array'
ORDER BY user_id
LIMIT $2
