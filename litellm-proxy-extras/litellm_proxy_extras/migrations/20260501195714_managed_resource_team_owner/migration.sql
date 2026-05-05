-- Adds `team_id` to managed-resource tables so service-account API
-- keys (no `user_id`) can be scoped by team instead of bypassing the
-- `created_by` filter entirely. Existing rows keep `team_id = NULL`
-- and become invisible to team-only callers — that is the intended isolation
-- outcome; backfill manually if legacy rows must remain visible.
--
-- The composite indexes match the listing query: filter by team owner, sort by
-- created_at DESC. Tables are typically small (resources per tenant, not per
-- request); a future operator with a large table can switch to
-- CREATE INDEX CONCURRENTLY in a follow-up migration.

ALTER TABLE "LiteLLM_ManagedFileTable" ADD COLUMN IF NOT EXISTS "team_id" TEXT;
ALTER TABLE "LiteLLM_ManagedObjectTable" ADD COLUMN IF NOT EXISTS "team_id" TEXT;
ALTER TABLE "LiteLLM_ManagedVectorStoreTable" ADD COLUMN IF NOT EXISTS "team_id" TEXT;

-- Index names follow Prisma's auto-generated convention so `prisma migrate diff`
-- against the schema is clean.
CREATE INDEX IF NOT EXISTS "LiteLLM_ManagedFileTable_team_id_created_at_idx" ON "LiteLLM_ManagedFileTable" ("team_id", "created_at" DESC);
CREATE INDEX IF NOT EXISTS "LiteLLM_ManagedObjectTable_team_id_created_at_idx" ON "LiteLLM_ManagedObjectTable" ("team_id", "created_at" DESC);
CREATE INDEX IF NOT EXISTS "LiteLLM_ManagedVectorStoreTable_team_id_created_at_idx" ON "LiteLLM_ManagedVectorStoreTable" ("team_id", "created_at" DESC);
