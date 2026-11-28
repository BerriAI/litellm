DELETE FROM "LiteLLM_TeamMembership"
WHERE user_id = $1
  AND team_id = $2
