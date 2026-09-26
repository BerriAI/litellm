INSERT INTO "LiteLLM_AutoRouterBaselineObservation"
    (request_id, scope, started_at, revision, data)
VALUES ($1, $2, $3::float8, $4::bigint, $5)
ON CONFLICT (request_id) DO NOTHING
