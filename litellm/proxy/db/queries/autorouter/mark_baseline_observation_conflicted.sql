UPDATE "LiteLLM_AutoRouterBaselineObservation"
SET conflicted = TRUE, revision = $4::bigint
WHERE request_id = $1 AND scope = $2 AND data <> $3 AND NOT conflicted
