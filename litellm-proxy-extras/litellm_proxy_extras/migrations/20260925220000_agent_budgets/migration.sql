ALTER TABLE "LiteLLM_AgentsTable" ADD COLUMN IF NOT EXISTS "budget_id" TEXT;

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_AgentsTable_budget_id_key" ON "LiteLLM_AgentsTable"("budget_id");

-- AddForeignKey
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_AgentsTable_budget_id_fkey') THEN
        ALTER TABLE "LiteLLM_AgentsTable" ADD CONSTRAINT "LiteLLM_AgentsTable_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE;
    END IF;
END $$;

