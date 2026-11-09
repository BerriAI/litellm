SELECT jsonb_build_object('user_id', user_id) AS "data!: AuthData" FROM "LiteLLM_EndUserTable"
WHERE blocked OR budget_id IS NOT NULL OR allowed_model_region IS NOT NULL OR default_model IS NOT NULL OR object_permission_id IS NOT NULL OR cardinality(models) > 0
ORDER BY user_id LIMIT $1
