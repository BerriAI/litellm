UPDATE "LiteLLM_AutoRouterBaselineObservation" AS observations
SET publication = x.publication::text
FROM jsonb_to_recordset($1::jsonb) AS x(request_id text, publication jsonb)
WHERE observations.request_id = x.request_id
