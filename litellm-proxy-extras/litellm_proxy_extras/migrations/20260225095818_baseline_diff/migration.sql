-- DropForeignKey
ALTER TABLE "LiteLLM_AgentsTable" DROP CONSTRAINT "LiteLLM_AgentsTable_object_permission_id_fkey";

-- AlterTable
ALTER TABLE "LiteLLM_AgentsTable" DROP COLUMN "object_permission_id";

-- AlterTable
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN     "spec_path" TEXT;

-- AlterTable
ALTER TABLE "LiteLLM_VerificationToken" DROP COLUMN "agent_id";

-- DropTable
DROP TABLE "LiteLLM_ToolTable";

