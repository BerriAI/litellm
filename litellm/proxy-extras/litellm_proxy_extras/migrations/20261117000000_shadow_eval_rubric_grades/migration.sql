SET LOCAL lock_timeout = '10s';

-- AlterTable
ALTER TABLE "LiteLLM_ShadowEvalJob" ADD COLUMN IF NOT EXISTS "rubric" TEXT;

-- AlterTable
ALTER TABLE "LiteLLM_ShadowEvalJob" ADD COLUMN IF NOT EXISTS "rubric_version" INTEGER;

-- AlterTable
ALTER TABLE "LiteLLM_ShadowEvalAttempt" ADD COLUMN IF NOT EXISTS "real_grade" INTEGER;

-- AlterTable
ALTER TABLE "LiteLLM_ShadowEvalAttempt" ADD COLUMN IF NOT EXISTS "shadow_grade" INTEGER;
