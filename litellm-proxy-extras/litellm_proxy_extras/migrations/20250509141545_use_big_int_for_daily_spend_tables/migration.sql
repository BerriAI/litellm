-- AlterTable
ALTER TABLE "LiteLLM_DailyTagSpend" ALTER COLUMN "prompt_tokens" SET DATA TYPE BIGINT,
ALTER COLUMN "completion_tokens" SET DATA TYPE BIGINT,
ALTER COLUMN "cache_read_input_tokens" SET DATA TYPE BIGINT,
ALTER COLUMN "cache_creation_input_tokens" SET DATA TYPE BIGINT,
ALTER COLUMN "api_requests" SET DATA TYPE BIGINT,
ALTER COLUMN "successful_requests" SET DATA TYPE BIGINT,
ALTER COLUMN "failed_requests" SET DATA TYPE BIGINT;

-- AlterTable
ALTER TABLE "LiteLLM_DailyTeamSpend" ALTER COLUMN "prompt_tokens" SET DATA TYPE BIGINT,
ALTER COLUMN "completion_tokens" SET DATA TYPE BIGINT,
ALTER COLUMN "api_requests" SET DATA TYPE BIGINT,
ALTER COLUMN "successful_requests" SET DATA TYPE BIGINT,
ALTER COLUMN "failed_requests" SET DATA TYPE BIGINT,
ALTER COLUMN "cache_creation_input_tokens" SET DATA TYPE BIGINT,
ALTER COLUMN "cache_read_input_tokens" SET DATA TYPE BIGINT;

-- AlterTable
ALTER TABLE "LiteLLM_DailyUserSpend" ALTER COLUMN "prompt_tokens" SET DATA TYPE BIGINT,
ALTER COLUMN "completion_tokens" SET DATA TYPE BIGINT,
ALTER COLUMN "api_requests" SET DATA TYPE BIGINT,
ALTER COLUMN "failed_requests" SET DATA TYPE BIGINT,
ALTER COLUMN "successful_requests" SET DATA TYPE BIGINT,
ALTER COLUMN "cache_creation_input_tokens" SET DATA TYPE BIGINT,
ALTER COLUMN "cache_read_input_tokens" SET DATA TYPE BIGINT;

