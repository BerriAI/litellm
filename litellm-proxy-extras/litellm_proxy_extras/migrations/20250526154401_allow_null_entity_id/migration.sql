-- AlterTable
ALTER TABLE "LiteLLM_DailyTagSpend" ALTER COLUMN "tag" DROP NOT NULL;

-- AlterTable
ALTER TABLE "LiteLLM_DailyTeamSpend" ALTER COLUMN "team_id" DROP NOT NULL;

-- AlterTable
ALTER TABLE "LiteLLM_DailyUserSpend" ALTER COLUMN "user_id" DROP NOT NULL;

