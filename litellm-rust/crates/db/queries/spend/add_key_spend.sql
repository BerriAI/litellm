WITH added AS (
    SELECT * FROM UNNEST($1::text[], $2::float8[]) AS a(id, cost)
), locked AS (
    SELECT t.token FROM "LiteLLM_VerificationToken" t
    WHERE t.token IN (SELECT id FROM added)
    ORDER BY t.token FOR UPDATE
)
UPDATE "LiteLLM_VerificationToken" t
SET spend = t.spend + added.cost,
    total_spend = t.total_spend + added.cost,
    last_active = (NOW() AT TIME ZONE 'UTC'),
    updated_at = (NOW() AT TIME ZONE 'UTC')
FROM added
WHERE t.token = added.id AND t.token IN (SELECT token FROM locked)
