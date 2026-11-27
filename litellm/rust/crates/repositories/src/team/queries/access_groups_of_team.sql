SELECT access_group_id,
       ($1 = ANY (COALESCE(assigned_team_ids, ARRAY[]::text[]))) IS TRUE AS "attached!"
FROM "LiteLLM_AccessGroupTable"
WHERE access_group_id = ANY ($2::text[])
   OR $1 = ANY (COALESCE(assigned_team_ids, ARRAY[]::text[]))
ORDER BY access_group_id
