WITH times AS (
    SELECT DISTINCT started_at FROM "LiteLLM_AutoRouterBaselineObservation"
    WHERE scope = $1 AND revision > $2::bigint
      AND ($3::float8 IS NULL OR started_at > $3::float8)
      AND ($5::float8 IS NULL OR (
          started_at >= $5::float8 AND publication::jsonb->>'status' = 'estimated'
      ))
    ORDER BY started_at LIMIT $4::int
)
SELECT data, publication, conflicted, started_at
FROM "LiteLLM_AutoRouterBaselineObservation"
WHERE scope = $1 AND revision > $2::bigint
  AND started_at IN (SELECT started_at FROM times)
  AND ($5::float8 IS NULL OR publication::jsonb->>'status' = 'estimated')
ORDER BY started_at, request_id
