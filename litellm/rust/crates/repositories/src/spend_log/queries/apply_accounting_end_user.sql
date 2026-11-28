INSERT INTO "LiteLLM_EndUserTable" (user_id, spend, blocked)
SELECT id, spend, false FROM jsonb_to_recordset($1::jsonb) AS input(id text, spend double precision)
ORDER BY id COLLATE "C"
ON CONFLICT (user_id) DO UPDATE
SET spend = "LiteLLM_EndUserTable".spend + EXCLUDED.spend
