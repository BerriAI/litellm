WITH locked AS MATERIALIZED (
    SELECT input.* FROM jsonb_to_recordset($1::jsonb) AS input(id text, member_id text, spend double precision)
    JOIN "LiteLLM_AppCustomerMemberTable" AS stored ON stored.app_id = input.id AND stored.customer_member_id = input.member_id
    ORDER BY input.id COLLATE "C", input.member_id COLLATE "C"
    FOR UPDATE OF stored
)
UPDATE "LiteLLM_AppCustomerMemberTable" AS stored
SET spend = stored.spend + input.spend
FROM locked AS input
WHERE stored.app_id = input.id AND stored.customer_member_id = input.member_id
