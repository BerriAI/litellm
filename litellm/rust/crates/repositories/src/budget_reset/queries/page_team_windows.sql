SELECT team_id AS id, budget_limits::jsonb AS "limits!: RawJson"
FROM "LiteLLM_TeamTable"
WHERE team_id > $1 AND jsonb_typeof(budget_limits::jsonb) = 'array'
ORDER BY team_id
LIMIT $2
