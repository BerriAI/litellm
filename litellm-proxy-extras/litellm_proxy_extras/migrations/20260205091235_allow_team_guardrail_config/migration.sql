-- AlterTable
ALTER TABLE "LiteLLM_DeletedTeamTable" ADD COLUMN     "allow_team_guardrail_config" BOOLEAN NOT NULL DEFAULT false;

-- AlterTable
ALTER TABLE "LiteLLM_TeamTable" ADD COLUMN     "allow_team_guardrail_config" BOOLEAN NOT NULL DEFAULT false;

