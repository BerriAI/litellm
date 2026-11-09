SELECT COALESCE(spend, 0.0)::float8 AS "spend!" FROM "LiteLLM_TagTable" WHERE tag_name=$1
