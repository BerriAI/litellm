SET LOCAL lock_timeout = '10s';

-- AlterTable
ALTER TABLE "LiteLLM_ShadowEvalJob" ADD COLUMN IF NOT EXISTS "skip_same_model" BOOLEAN NOT NULL DEFAULT false;

-- AlterTable
ALTER TABLE "LiteLLM_ShadowEvalFunnel" ADD COLUMN IF NOT EXISTS "same_model" INTEGER NOT NULL DEFAULT 0;
