WITH changes AS (
    SELECT request_id, publication::jsonb AS publication
    FROM jsonb_to_recordset($1::jsonb) AS x(request_id text, publication jsonb)
)
UPDATE "LiteLLM_SpendLogs" AS logs
SET metadata = (COALESCE(logs.metadata::jsonb, '{}'::jsonb) - 'autorouter_baseline_observation') || jsonb_build_object(
    'autorouter_savings_estimate', changes.publication,
    'autorouter_savings', CASE WHEN changes.publication->>'status' = 'estimated' THEN
        (changes.publication->>'baseline_spend')::float8 - (changes.publication->>'actual_spend')::float8
        ELSE NULL END
)
FROM changes WHERE logs.request_id = changes.request_id
