WITH expired AS (
    SELECT scope FROM "LiteLLM_AutoRouterBaselineComparison"
    WHERE NOT retired AND updated_at < $1::timestamptz
    ORDER BY updated_at LIMIT $2::int FOR UPDATE SKIP LOCKED
)
UPDATE "LiteLLM_AutoRouterBaselineComparison" AS comparison
SET retired = TRUE, history = NULL
FROM expired WHERE comparison.scope = expired.scope
