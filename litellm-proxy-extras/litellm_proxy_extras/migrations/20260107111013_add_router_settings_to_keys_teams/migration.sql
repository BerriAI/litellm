-- AlterTable
ALTER TABLE "LiteLLM_TeamTable" ADD COLUMN     "router_settings" JSONB DEFAULT '{}';

-- AlterTable
ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN     "router_settings" JSONB DEFAULT '{}';

