INSERT INTO "LiteLLM_DailyGuardrailUsageUnits" AS stored (guardrail_id, date, team_id, api_key, usage_unit, units, cost, untracked_units, updated_at)
SELECT guardrail_id, date, team_id, api_key, usage_unit, units, cost, untracked_units, (NOW() AT TIME ZONE 'UTC')
FROM jsonb_to_recordset($1::jsonb) AS input(
    guardrail_id text,
    date text,
    team_id text,
    api_key text,
    usage_unit text,
    units bigint,
    cost double precision,
    untracked_units bigint
)
ORDER BY guardrail_id COLLATE "C", date COLLATE "C", team_id COLLATE "C", api_key COLLATE "C", usage_unit COLLATE "C"
ON CONFLICT (guardrail_id, date, team_id, api_key, usage_unit) DO UPDATE SET
    units = stored.units + EXCLUDED.units,
    cost = stored.cost + EXCLUDED.cost,
    untracked_units = stored.untracked_units + EXCLUDED.untracked_units,
    updated_at = EXCLUDED.updated_at
