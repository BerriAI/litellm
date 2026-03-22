-- AlterTable
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN     "team_id" TEXT;

-- AlterTable
ALTER TABLE "LiteLLM_ObjectPermissionTable" ADD COLUMN     "search_tools" TEXT[] DEFAULT ARRAY['*']::TEXT[];

