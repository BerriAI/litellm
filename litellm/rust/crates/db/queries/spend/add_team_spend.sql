WITH added AS (
    SELECT id, SUM(cost) AS cost
    FROM UNNEST($1::text[], $2::float8[]) AS a(id, cost)
    GROUP BY id
),
locked AS (
    SELECT t.team_id FROM "LiteLLM_TeamTable" t
    WHERE t.team_id IN (SELECT id FROM added)
    ORDER BY t.team_id FOR UPDATE
)
UPDATE "LiteLLM_TeamTable" t
SET spend = t.spend + added.cost,
    updated_at = (NOW() AT TIME ZONE 'UTC')
FROM added
WHERE t.team_id = added.id AND t.team_id IN (SELECT team_id FROM locked)
