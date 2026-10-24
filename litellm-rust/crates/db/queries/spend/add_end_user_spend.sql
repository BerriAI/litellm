INSERT INTO "LiteLLM_EndUserTable" (user_id, spend, blocked)
SELECT id, cost, false FROM UNNEST($1::text[], $2::float8[]) AS a(id, cost)
ORDER BY id
ON CONFLICT (user_id) DO UPDATE
SET spend = "LiteLLM_EndUserTable".spend + EXCLUDED.spend
