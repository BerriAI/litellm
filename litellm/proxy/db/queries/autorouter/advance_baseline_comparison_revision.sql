UPDATE "LiteLLM_AutoRouterBaselineComparison"
SET revision = $2::bigint, updated_at = CURRENT_TIMESTAMP, attempted_at = NULL
WHERE scope = $1
