UPDATE "LiteLLM_AccessGroupTable"
SET assigned_team_ids = array_remove(assigned_team_ids, $1)
WHERE $1 = ANY (COALESCE(assigned_team_ids, ARRAY[]::text[]))
  AND access_group_id = ANY ($2::text[])
