WITH locked AS MATERIALIZED (
    SELECT input.* FROM jsonb_to_recordset($1::jsonb) AS input(id text, spend double precision)
    JOIN "LiteLLM_AppGroupTable" AS stored ON stored.app_group_id = input.id
    ORDER BY input.id COLLATE "C"
    FOR UPDATE OF stored
)
UPDATE "LiteLLM_AppGroupTable" AS stored
SET spend = stored.spend + input.spend,
    updated_at = (NOW() AT TIME ZONE 'UTC')
FROM locked AS input
WHERE stored.app_group_id = input.id
