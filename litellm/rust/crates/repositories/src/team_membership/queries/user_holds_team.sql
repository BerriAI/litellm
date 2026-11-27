SELECT $2 = ANY (COALESCE(teams, '{}')) AS "holds!"
FROM "LiteLLM_UserTable"
WHERE user_id = $1
