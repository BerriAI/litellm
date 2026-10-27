INSERT INTO "LiteLLM_EndUserTable" (user_id, spend, blocked)
SELECT id, SUM(cost), false FROM UNNEST($1::text[], $2::float8[]) AS a(id, cost)
GROUP BY id
ORDER BY id
ON CONFLICT (user_id) DO UPDATE
SET spend = "LiteLLM_EndUserTable".spend + EXCLUDED.spend
