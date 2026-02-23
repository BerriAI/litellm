-- DropForeignKey
ALTER TABLE "LiteLLM_TeamMembership" DROP CONSTRAINT "LiteLLM_TeamMembership_budget_id_fkey";

-- AlterTable
ALTER TABLE "LiteLLM_ManagedFileTable" ALTER COLUMN "file_object" DROP NOT NULL;

-- AddForeignKey
ALTER TABLE "LiteLLM_TeamMembership" ADD CONSTRAINT "LiteLLM_TeamMembership_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE;

