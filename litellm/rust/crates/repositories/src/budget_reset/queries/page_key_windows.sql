SELECT token AS id, budget_limits::jsonb AS "limits!: RawJson"
FROM "LiteLLM_VerificationToken"
WHERE token > $1 AND jsonb_typeof(budget_limits::jsonb) = 'array'
ORDER BY token
LIMIT $2
