SELECT COALESCE(spend, 0.0)::float8 AS "spend!" FROM "LiteLLM_ProjectTable" WHERE project_id=$1
