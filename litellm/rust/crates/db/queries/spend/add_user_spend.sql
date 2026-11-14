WITH added AS (
    SELECT id, SUM(cost) AS cost
    FROM UNNEST($1::text[], $2::float8[]) AS a(id, cost)
    GROUP BY id
), locked AS (
    SELECT u.user_id FROM "LiteLLM_UserTable" u
    WHERE u.user_id IN (SELECT id FROM added)
    ORDER BY u.user_id FOR UPDATE
)
UPDATE "LiteLLM_UserTable" u
SET spend = u.spend + added.cost,
    updated_at = (NOW() AT TIME ZONE 'UTC')
FROM added
WHERE u.user_id = added.id AND u.user_id IN (SELECT user_id FROM locked)
