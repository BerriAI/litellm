-- Fix duplicate-key issue on LiteLLM_MemoryTable.
--
-- Problem: the unique constraint `(key, user_id, team_id)` does not block
-- duplicates when `user_id` or `team_id` is NULL, because by default
-- Postgres treats each NULL as distinct (ANSI SQL semantics). Callers with
-- no team_id could POST the same key repeatedly and get multiple rows.
--
-- Fix (Postgres 15+): recreate the unique index with NULLS NOT DISTINCT so
-- NULL values are treated as equal for uniqueness checks.
--
-- Note: Prisma has no schema syntax for NULLS NOT DISTINCT (as of writing),
-- so this lives as a raw SQL migration. Prisma introspection will still see
-- the index as uniquely covering `(key, user_id, team_id)`, matching the
-- `@@unique` in schema.prisma.

-- 1. Deduplicate existing rows: keep only the most-recently-updated row per
--    (key, user_id, team_id), using IS NOT DISTINCT FROM so NULL == NULL.
DELETE FROM "LiteLLM_MemoryTable" a
USING "LiteLLM_MemoryTable" b
WHERE a.memory_id <> b.memory_id
  AND a.key = b.key
  AND a.user_id IS NOT DISTINCT FROM b.user_id
  AND a.team_id IS NOT DISTINCT FROM b.team_id
  AND (
    a.updated_at < b.updated_at
    OR (a.updated_at = b.updated_at AND a.memory_id < b.memory_id)
  );

-- 2. Drop the old (NULL-distinct) unique index.
DROP INDEX IF EXISTS "LiteLLM_MemoryTable_key_user_id_team_id_key";

-- 3. Recreate with NULLS NOT DISTINCT.
CREATE UNIQUE INDEX "LiteLLM_MemoryTable_key_user_id_team_id_key"
    ON "LiteLLM_MemoryTable"("key", "user_id", "team_id") NULLS NOT DISTINCT;
