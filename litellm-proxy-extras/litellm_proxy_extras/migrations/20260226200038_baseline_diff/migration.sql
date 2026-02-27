-- AlterTable
ALTER TABLE "LiteLLM_DeletedVerificationToken" DROP COLUMN "agent_id";

-- AlterTable
ALTER TABLE "LiteLLM_ObjectPermissionTable" DROP COLUMN "blocked_tools";

-- DropTable
DROP TABLE "LiteLLM_SpendLogToolIndex";

