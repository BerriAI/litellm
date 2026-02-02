-- AlterTable
ALTER TABLE "LiteLLM_DeletedTeamTable" ADD COLUMN     "router_settings" JSONB DEFAULT '{}';

-- AlterTable
ALTER TABLE "LiteLLM_DeletedVerificationToken" ADD COLUMN     "router_settings" JSONB DEFAULT '{}';

