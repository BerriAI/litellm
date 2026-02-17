-- AlterTable
ALTER TABLE "LiteLLM_ManagedVectorStoresTable" ADD COLUMN     "team_id" TEXT,
ADD COLUMN     "user_id" TEXT;

-- CreateIndex
CREATE INDEX "LiteLLM_ManagedVectorStoresTable_team_id_idx" ON "LiteLLM_ManagedVectorStoresTable"("team_id");

-- CreateIndex
CREATE INDEX "LiteLLM_ManagedVectorStoresTable_user_id_idx" ON "LiteLLM_ManagedVectorStoresTable"("user_id");

