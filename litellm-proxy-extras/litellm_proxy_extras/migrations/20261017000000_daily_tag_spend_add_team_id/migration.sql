-- Records the team that sent each request on its daily tag spend rows, so one tag used by
-- several teams lands in one row per team. "" means the request had no team, and every row
-- written before this migration keeps "" (attribution is not retroactive).
--
-- NOT NULL DEFAULT '' rather than nullable: the daily spend upsert normalizes every conflict
-- column to '' because a NULL never matches itself in a unique index. On PostgreSQL 11+ adding
-- a column with a constant default is a metadata-only change, so this does not rewrite the table.
ALTER TABLE "LiteLLM_DailyTagSpend" ADD COLUMN IF NOT EXISTS "team_id" TEXT NOT NULL DEFAULT '';
