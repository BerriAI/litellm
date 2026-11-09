SELECT jsonb_build_object('tag_name', tag_name) AS "data!: AuthData" FROM "LiteLLM_TagTable" ORDER BY tag_name LIMIT $1
