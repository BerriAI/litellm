-- AlterTable
ALTER TABLE "LiteLLM_TagTable" ADD COLUMN IF NOT EXISTS "team_id" TEXT;

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_TagTable_team_id_idx" ON "LiteLLM_TagTable"("team_id");

-- AddForeignKey
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_TagTable_team_id_fkey') THEN
        ALTER TABLE "LiteLLM_TagTable" ADD CONSTRAINT "LiteLLM_TagTable_team_id_fkey" FOREIGN KEY ("team_id") REFERENCES "LiteLLM_TeamTable"("team_id") ON DELETE SET NULL ON UPDATE CASCADE;
    END IF;
END $$;
