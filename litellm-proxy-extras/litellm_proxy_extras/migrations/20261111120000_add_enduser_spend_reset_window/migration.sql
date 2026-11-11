SET LOCAL lock_timeout = '10s';

-- AlterTable
ALTER TABLE "LiteLLM_EndUserTable" ADD COLUMN IF NOT EXISTS "spend_reset_window" TIMESTAMP(3);
