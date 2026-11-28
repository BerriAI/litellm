SELECT members_with_roles AS "members_with_roles!: RawJson"
FROM "LiteLLM_TeamTable"
WHERE team_id = $1
