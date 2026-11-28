UPDATE "LiteLLM_UserTable"
SET teams = array_remove(teams, $2),
    updated_at = NOW() AT TIME ZONE 'UTC'
WHERE user_id = $1
  AND $2 = ANY (teams)
