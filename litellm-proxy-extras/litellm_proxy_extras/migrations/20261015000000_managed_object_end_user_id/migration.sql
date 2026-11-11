SET LOCAL lock_timeout = '10s';

-- AlterTable
ALTER TABLE "LiteLLM_ManagedObjectTable" ADD COLUMN IF NOT EXISTS "end_user_id" TEXT;
