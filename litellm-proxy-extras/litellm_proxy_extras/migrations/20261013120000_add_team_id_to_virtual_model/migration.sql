SET LOCAL lock_timeout = '10s';
-- AlterTable
ALTER TABLE "LiteLLM_VirtualModelTable" ADD COLUMN IF NOT EXISTS "team_id" TEXT;
