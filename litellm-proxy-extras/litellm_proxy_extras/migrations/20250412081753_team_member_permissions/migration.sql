-- AlterTable
ALTER TABLE "LiteLLM_TeamTable" ADD COLUMN     "team_member_permissions" TEXT[] DEFAULT ARRAY[]::TEXT[];

