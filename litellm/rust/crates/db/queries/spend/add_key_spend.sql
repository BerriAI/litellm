WITH added AS (
    SELECT id, SUM(cost) AS cost
    FROM UNNEST($1::text[], $2::float8[]) AS a(id, cost)
    GROUP BY id
), locked AS (
    SELECT t.token FROM "LiteLLM_VerificationToken" t
    WHERE t.token IN (SELECT id FROM added)
    ORDER BY t.token FOR UPDATE
)
UPDATE "LiteLLM_VerificationToken" t
SET spend = t.spend + added.cost,
    total_spend = t.total_spend + added.cost,
    last_active = (clock_timestamp() AT TIME ZONE 'UTC'),
    updated_at = (clock_timestamp() AT TIME ZONE 'UTC')
FROM added
WHERE t.token = added.id AND t.token IN (SELECT token FROM locked)
