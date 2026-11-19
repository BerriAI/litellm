SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_AppCustomerOrgTable"
  ADD COLUMN "budget_id" TEXT,
  ADD COLUMN "budget_override_expires_at" TIMESTAMP(3);

ALTER TABLE "LiteLLM_AppCustomerMemberTable"
  ADD COLUMN "budget_id" TEXT,
  ADD COLUMN "budget_override_expires_at" TIMESTAMP(3);

ALTER TABLE "LiteLLM_AppCustomerOrgTable"
  ADD CONSTRAINT "LiteLLM_AppCustomerOrgTable_budget_id_fkey"
  FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE RESTRICT ON UPDATE CASCADE;

ALTER TABLE "LiteLLM_AppCustomerMemberTable"
  ADD CONSTRAINT "LiteLLM_AppCustomerMemberTable_budget_id_fkey"
  FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE RESTRICT ON UPDATE CASCADE;
