SELECT COALESCE(spend, 0.0)::float8 AS "spend!" FROM "LiteLLM_UserTable" WHERE user_id=$1
