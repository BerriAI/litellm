-- AlterTable
ALTER TABLE "LiteLLM_DailyUserSpend" ADD COLUMN IF NOT EXISTS "compression_saved_tokens" BIGINT NOT NULL DEFAULT 0;

-- AlterTable
ALTER TABLE "LiteLLM_DailyOrganizationSpend" ADD COLUMN IF NOT EXISTS "compression_saved_tokens" BIGINT NOT NULL DEFAULT 0;

-- AlterTable
ALTER TABLE "LiteLLM_DailyEndUserSpend" ADD COLUMN IF NOT EXISTS "compression_saved_tokens" BIGINT NOT NULL DEFAULT 0;

-- AlterTable
ALTER TABLE "LiteLLM_DailyAgentSpend" ADD COLUMN IF NOT EXISTS "compression_saved_tokens" BIGINT NOT NULL DEFAULT 0;

-- AlterTable
ALTER TABLE "LiteLLM_DailyTeamSpend" ADD COLUMN IF NOT EXISTS "compression_saved_tokens" BIGINT NOT NULL DEFAULT 0;

-- AlterTable
ALTER TABLE "LiteLLM_DailyTagSpend" ADD COLUMN IF NOT EXISTS "compression_saved_tokens" BIGINT NOT NULL DEFAULT 0;
