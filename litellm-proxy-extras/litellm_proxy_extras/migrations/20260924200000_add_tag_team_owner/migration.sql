-- AlterTable
ALTER TABLE "LiteLLM_TagTable" ADD COLUMN IF NOT EXISTS "team_id" TEXT;

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_TagTable_team_id_idx" ON "LiteLLM_TagTable"("team_id");
