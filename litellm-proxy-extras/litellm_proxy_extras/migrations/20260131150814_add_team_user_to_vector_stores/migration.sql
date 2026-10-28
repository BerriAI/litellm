-- AlterTable
ALTER TABLE "LiteLLM_ManagedVectorStoresTable"
    ADD COLUMN IF NOT EXISTS "team_id" TEXT,
    ADD COLUMN IF NOT EXISTS "user_id" TEXT;

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_ManagedVectorStoresTable_team_id_idx"
    ON "LiteLLM_ManagedVectorStoresTable"("team_id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_ManagedVectorStoresTable_user_id_idx"
    ON "LiteLLM_ManagedVectorStoresTable"("user_id");

