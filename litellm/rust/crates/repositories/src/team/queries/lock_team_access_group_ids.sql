SELECT COALESCE(access_group_ids, '{}') AS "access_group_ids!"
FROM "LiteLLM_TeamTable"
WHERE team_id = $1
FOR UPDATE
