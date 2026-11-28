WITH locked AS MATERIALIZED (
    SELECT input.* FROM jsonb_to_recordset($1::jsonb) AS input(id text, spend double precision)
    JOIN "LiteLLM_TagTable" AS stored ON stored.tag_name = input.id
    ORDER BY input.id COLLATE "C"
    FOR UPDATE OF stored
)
UPDATE "LiteLLM_TagTable" AS stored
SET spend = stored.spend + input.spend,
    updated_at = (NOW() AT TIME ZONE 'UTC')
FROM locked AS input
WHERE stored.tag_name = input.id
