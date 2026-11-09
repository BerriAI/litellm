SELECT COALESCE(spend, 0.0)::float8 AS "spend!" FROM "LiteLLM_TeamTable" WHERE team_id=$1
