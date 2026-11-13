-- AlterTable
ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN     "budget_limits" JSONB,
ADD COLUMN     "cap_policy" TEXT,
ADD COLUMN     "is_profile" BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN     "temp_budget_duration" TEXT;

-- AlterTable
ALTER TABLE "LiteLLM_UserTable" ADD COLUMN     "budget_id" TEXT,
ADD COLUMN     "budget_override_expires_at" TIMESTAMP(3),
ADD COLUMN     "budget_override_id" TEXT;

-- AlterTable
ALTER TABLE "LiteLLM_ManagedObjectTable" ADD COLUMN     "budget_attribution" JSONB;

-- CreateIndex
CREATE INDEX "LiteLLM_UserTable_budget_id_idx" ON "LiteLLM_UserTable"("budget_id");

-- CreateIndex
CREATE INDEX "LiteLLM_UserTable_budget_override_id_idx" ON "LiteLLM_UserTable"("budget_override_id");

-- AddForeignKey
ALTER TABLE "LiteLLM_UserTable" ADD CONSTRAINT "LiteLLM_UserTable_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_UserTable" ADD CONSTRAINT "LiteLLM_UserTable_budget_override_id_fkey" FOREIGN KEY ("budget_override_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE RESTRICT ON UPDATE CASCADE;

