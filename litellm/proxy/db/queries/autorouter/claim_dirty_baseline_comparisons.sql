WITH candidates AS (
    SELECT scope FROM "LiteLLM_AutoRouterBaselineComparison"
    WHERE NOT retired AND revision <> published_revision
      AND (attempted_at IS NULL OR attempted_at < CURRENT_TIMESTAMP - INTERVAL '30 seconds')
    ORDER BY attempted_at NULLS FIRST, updated_at, scope LIMIT 32 FOR UPDATE SKIP LOCKED
)
UPDATE "LiteLLM_AutoRouterBaselineComparison" AS comparison
SET attempted_at = CURRENT_TIMESTAMP FROM candidates
WHERE comparison.scope = candidates.scope RETURNING comparison.scope
