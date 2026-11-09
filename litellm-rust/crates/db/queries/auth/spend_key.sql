SELECT COALESCE(spend, 0.0)::float8 AS "spend!" FROM "LiteLLM_VerificationToken" WHERE token=$1
