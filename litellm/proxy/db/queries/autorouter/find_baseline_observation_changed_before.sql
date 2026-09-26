SELECT 1 AS found FROM "LiteLLM_AutoRouterBaselineObservation"
WHERE scope = $1 AND revision > $2::bigint AND started_at <= $3::float8
LIMIT 1
