-- DropForeignKey
ALTER TABLE "LiteLLM_TeamMembership" DROP CONSTRAINT "LiteLLM_TeamMembership_budget_id_fkey";

-- AddForeignKey
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_TeamMembership_budget_id_fkey') THEN
        ALTER TABLE "LiteLLM_TeamMembership" ADD CONSTRAINT "LiteLLM_TeamMembership_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;

