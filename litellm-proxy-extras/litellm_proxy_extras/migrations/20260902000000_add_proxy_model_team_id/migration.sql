-- AlterTable
ALTER TABLE "LiteLLM_ProxyModelTable" ADD COLUMN IF NOT EXISTS "team_id" TEXT;

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_ProxyModelTable_team_id_idx" ON "LiteLLM_ProxyModelTable"("team_id");
