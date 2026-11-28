WITH locked AS MATERIALIZED (
    SELECT input.* FROM jsonb_to_recordset($1::jsonb) AS input(id text, member_id text, spend double precision)
    JOIN "LiteLLM_OrganizationMembership" AS stored ON stored.organization_id = input.id AND stored.user_id = input.member_id
    ORDER BY input.id COLLATE "C", input.member_id COLLATE "C"
    FOR UPDATE OF stored
)
UPDATE "LiteLLM_OrganizationMembership" AS stored
SET spend = stored.spend + input.spend,
    updated_at = (NOW() AT TIME ZONE 'UTC')
FROM locked AS input
WHERE stored.organization_id = input.id AND stored.user_id = input.member_id
