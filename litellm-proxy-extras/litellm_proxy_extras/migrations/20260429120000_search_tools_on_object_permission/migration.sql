-- Search tool allowlists live on LiteLLM_ObjectPermissionTable (with agents, MCP, vector stores).
ALTER TABLE "LiteLLM_ObjectPermissionTable" ADD COLUMN IF NOT EXISTS "search_tools" TEXT[] DEFAULT ARRAY[]::TEXT[];

-- Unshipped columns: drop if present (e.g. local DBs that had previous Prisma migrate).
ALTER TABLE "LiteLLM_TeamTable" DROP COLUMN IF EXISTS "allowed_search_tools";
ALTER TABLE "LiteLLM_VerificationToken" DROP COLUMN IF EXISTS "allowed_search_tools";
