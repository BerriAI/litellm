SET LOCAL lock_timeout = '10s';

-- AlterTable
ALTER TABLE "LiteLLM_AccessGroupTable" ADD COLUMN IF NOT EXISTS "access_skill_names" TEXT[] DEFAULT ARRAY[]::TEXT[];
