-- DropForeignKey
ALTER TABLE "LiteLLM_TeamMembership" DROP CONSTRAINT "LiteLLM_TeamMembership_budget_id_fkey";

-- AddForeignKey
ALTER TABLE "LiteLLM_TeamMembership" ADD CONSTRAINT "LiteLLM_TeamMembership_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE CASCADE ON UPDATE CASCADE;

