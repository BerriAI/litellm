SELECT COALESCE(spend, 0.0)::float8 AS "spend!" FROM "LiteLLM_TeamMembership" WHERE user_id=$1 AND team_id=$2
