-- AlterTable
ALTER TABLE "LiteLLM_AgentsTable" ADD COLUMN IF NOT EXISTS "kill_switch" JSONB;
