SELECT jsonb_build_object('spend', COALESCE(SUM(spend), 0.0)) AS "data!: AuthData" FROM "LiteLLM_SpendLogs" WHERE "api_key"=$1 AND "startTime" >= $2
