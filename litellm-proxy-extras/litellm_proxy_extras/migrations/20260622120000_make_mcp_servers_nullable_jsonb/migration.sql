-- AlterTable
-- Convert object_permission.mcp_servers from a non-nullable String[] to a nullable
-- JSONB column so an explicit empty list ([]) can mean "no servers" distinctly from
-- NULL ("inherit team scope"). Prisma's auto-diff would DROP/ADD the column (data
-- loss); this in-place USING cast preserves data. Existing empty arrays meant
-- "inherit" under the old semantics, so they map to NULL (which also means "inherit"
-- under the new semantics) -- no live key or team changes behavior.
ALTER TABLE "LiteLLM_ObjectPermissionTable"
  ALTER COLUMN "mcp_servers" DROP DEFAULT,
  ALTER COLUMN "mcp_servers" TYPE JSONB USING (
    CASE
      WHEN "mcp_servers" IS NULL OR cardinality("mcp_servers") = 0 THEN NULL
      ELSE to_jsonb("mcp_servers")
    END
  );
