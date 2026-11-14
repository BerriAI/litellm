-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_ModelAccessGroupBudgetTable" (
    "access_group_name" TEXT NOT NULL,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "budget_id" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT,

    CONSTRAINT "LiteLLM_ModelAccessGroupBudgetTable_pkey" PRIMARY KEY ("access_group_name")
);

-- AddForeignKey
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_ModelAccessGroupBudgetTable_budget_id_fkey') THEN
        ALTER TABLE "LiteLLM_ModelAccessGroupBudgetTable" ADD CONSTRAINT "LiteLLM_ModelAccessGroupBudgetTable_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE;
    END IF;
END $$;
