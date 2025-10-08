-- AlterTable
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN     "args" TEXT[] DEFAULT ARRAY[]::TEXT[],
ADD COLUMN     "command" TEXT,
ADD COLUMN     "env" JSONB DEFAULT '{}',
ADD COLUMN     "mcp_access_groups" TEXT[],
ALTER COLUMN "url" DROP NOT NULL;

-- AlterTable
ALTER TABLE "LiteLLM_ObjectPermissionTable" ADD COLUMN     "mcp_access_groups" TEXT[] DEFAULT ARRAY[]::TEXT[];

