WITH locked AS MATERIALIZED (
    SELECT input.* FROM jsonb_to_recordset($1::jsonb) AS input(id text, spend double precision)
    JOIN "LiteLLM_VerificationToken" AS stored ON stored.token = input.id
    ORDER BY input.id COLLATE "C"
    FOR UPDATE OF stored
)
UPDATE "LiteLLM_VerificationToken" AS stored
SET spend = stored.spend + input.spend,
    total_spend = stored.total_spend + input.spend,
    last_active = (NOW() AT TIME ZONE 'UTC'),
    updated_at = (NOW() AT TIME ZONE 'UTC')
FROM locked AS input
WHERE stored.token = input.id
