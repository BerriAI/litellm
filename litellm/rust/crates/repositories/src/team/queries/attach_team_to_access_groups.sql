UPDATE "LiteLLM_AccessGroupTable"
SET assigned_team_ids = array_append(COALESCE(assigned_team_ids, ARRAY[]::text[]), $1)
WHERE access_group_id = ANY ($2::text[])
  AND NOT ($1 = ANY (COALESCE(assigned_team_ids, ARRAY[]::text[])))
