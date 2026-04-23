-- DropForeignKey
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_TeamMembership_budget_id_fkey') THEN
        ALTER TABLE "LiteLLM_TeamMembership" DROP CONSTRAINT "LiteLLM_TeamMembership_budget_id_fkey";
    END IF;
END $$;

-- AlterTable
ALTER TABLE "LiteLLM_ManagedFileTable" ALTER COLUMN "file_object" DROP NOT NULL;

-- AddForeignKey
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_TeamMembership_budget_id_fkey') THEN
        ALTER TABLE "LiteLLM_TeamMembership" ADD CONSTRAINT "LiteLLM_TeamMembership_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE;
    END IF;
END $$;

