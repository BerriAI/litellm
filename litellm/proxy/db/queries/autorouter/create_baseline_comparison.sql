INSERT INTO "LiteLLM_AutoRouterBaselineComparison"
    (scope, api_key, session_id, router_name, initial_equivalent)
VALUES ($1, $2, $3, $4, NOT EXISTS (
    SELECT 1 FROM "LiteLLM_AutoRouterSession"
    WHERE api_key = $2 AND session_id = $3 AND router_name = $4
)) ON CONFLICT (scope) DO NOTHING
