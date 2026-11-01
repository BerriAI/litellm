SET LOCAL lock_timeout = '10s';

-- The team that owns the virtual key a tag row was spent through. "" when the request did not
-- come through a virtual key with a team (JWT, custom auth, OAuth2, team-less keys), and on every
-- row written before this migration. Not part of the unique key: a team's virtual key already
-- maps the row to one team, so older proxies keep upserting on the unchanged key during a rolling
-- upgrade. On PostgreSQL 11+ adding a column with a constant default does not rewrite the table.
ALTER TABLE "LiteLLM_DailyTagSpend" ADD COLUMN IF NOT EXISTS "team_id" TEXT NOT NULL DEFAULT '';
