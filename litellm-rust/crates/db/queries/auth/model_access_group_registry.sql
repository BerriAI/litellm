SELECT jsonb_build_object('access_group_name', access_group_name) AS "data!: AuthData" FROM "LiteLLM_ModelAccessGroupBudgetTable" ORDER BY access_group_name LIMIT $1
