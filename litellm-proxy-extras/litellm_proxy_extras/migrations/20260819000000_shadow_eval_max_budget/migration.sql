-- AlterTable
ALTER TABLE "LiteLLM_ShadowEvalJob" ADD COLUMN     "max_budget" DOUBLE PRECISION;

-- AlterTable
ALTER TABLE "LiteLLM_ShadowEvalAttempt" ADD COLUMN     "shadow_cost" DOUBLE PRECISION NOT NULL DEFAULT 0;
