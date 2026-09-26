SELECT 1 AS found FROM "LiteLLM_AutoRouterBaselineObservation" AS observation
WHERE scope = $1 AND publication IS NULL AND NOT EXISTS (
    SELECT 1 FROM "LiteLLM_SpendLogs" AS log WHERE log.request_id = observation.request_id
)
LIMIT 1
