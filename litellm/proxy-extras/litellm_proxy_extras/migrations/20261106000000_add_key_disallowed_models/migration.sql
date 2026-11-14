SET LOCAL lock_timeout = '10s';

-- AlterTable
ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN IF NOT EXISTS "disallowed_models" TEXT[] DEFAULT ARRAY[]::TEXT[];

-- AlterTable
ALTER TABLE "LiteLLM_DeletedVerificationToken" ADD COLUMN IF NOT EXISTS "disallowed_models" TEXT[] DEFAULT ARRAY[]::TEXT[];
