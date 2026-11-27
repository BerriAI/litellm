SELECT access_group_id
FROM "LiteLLM_AccessGroupTable"
WHERE access_group_id = ANY ($1::text[])
ORDER BY access_group_id COLLATE "C"
FOR UPDATE
