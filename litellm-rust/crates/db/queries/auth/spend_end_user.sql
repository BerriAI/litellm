SELECT COALESCE(spend, 0.0)::float8 AS "spend!" FROM "LiteLLM_EndUserTable" WHERE user_id=$1
