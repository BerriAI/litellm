SELECT data, publication, conflicted, started_at
FROM "LiteLLM_AutoRouterBaselineObservation"
WHERE request_id = $1 AND scope = $2
