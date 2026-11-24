SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_AppCustomerOrgTable"
  ADD COLUMN IF NOT EXISTS "budget_id" TEXT,
  ADD COLUMN IF NOT EXISTS "budget_override_expires_at" TIMESTAMP(3);

ALTER TABLE "LiteLLM_AppCustomerMemberTable"
  ADD COLUMN IF NOT EXISTS "budget_id" TEXT,
  ADD COLUMN IF NOT EXISTS "budget_override_expires_at" TIMESTAMP(3);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_AppCustomerOrgTable_budget_id_fkey') THEN
        ALTER TABLE "LiteLLM_AppCustomerOrgTable" ADD CONSTRAINT "LiteLLM_AppCustomerOrgTable_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE RESTRICT ON UPDATE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_AppCustomerMemberTable_budget_id_fkey') THEN
        ALTER TABLE "LiteLLM_AppCustomerMemberTable" ADD CONSTRAINT "LiteLLM_AppCustomerMemberTable_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE RESTRICT ON UPDATE CASCADE;
    END IF;
END $$;
