UPDATE "LiteLLM_UserTable"
SET teams = array_append(COALESCE(teams, '{}'), $2),
    updated_at = NOW() AT TIME ZONE 'UTC'
WHERE user_id = $1
  AND NOT ($2 = ANY (COALESCE(teams, '{}')))
