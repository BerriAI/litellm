SET LOCAL lock_timeout = '10s';

CREATE TABLE IF NOT EXISTS "LiteLLM_AppTable" (
    "app_id" TEXT NOT NULL,
    "alias" TEXT NOT NULL,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "spend_reset_window" TEXT,
    "budget_id" TEXT REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE,
    "default_tier_budget_id" TEXT REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE,
    "default_customer_member_budget_id" TEXT REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT NOT NULL,
    "updated_by" TEXT NOT NULL,
    CONSTRAINT "LiteLLM_AppTable_pkey" PRIMARY KEY ("app_id")
);
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_AppTable_alias_key" ON "LiteLLM_AppTable"("alias");

CREATE TABLE IF NOT EXISTS "LiteLLM_AppCustomerOrgTable" (
    "app_id" TEXT NOT NULL REFERENCES "LiteLLM_AppTable"("app_id") ON DELETE CASCADE ON UPDATE CASCADE,
    "customer_org_id" TEXT NOT NULL REFERENCES "LiteLLM_CustomerOrganizationTable"("customer_org_id") ON DELETE CASCADE ON UPDATE CASCADE,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "spend_reset_window" TEXT,
    "tier_budget_id" TEXT REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE,
    "budget_override_id" TEXT REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE,
    "customer_member_budget_id" TEXT REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT "LiteLLM_AppCustomerOrgTable_pkey" PRIMARY KEY ("app_id", "customer_org_id")
);

CREATE TABLE IF NOT EXISTS "LiteLLM_AppCustomerMemberTable" (
    "app_id" TEXT NOT NULL REFERENCES "LiteLLM_AppTable"("app_id") ON DELETE CASCADE ON UPDATE CASCADE,
    "customer_member_id" TEXT NOT NULL REFERENCES "LiteLLM_CustomerMemberTable"("customer_member_id") ON DELETE CASCADE ON UPDATE CASCADE,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "spend_reset_window" TEXT,
    "budget_override_id" TEXT REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT "LiteLLM_AppCustomerMemberTable_pkey" PRIMARY KEY ("app_id", "customer_member_id")
);

ALTER TABLE "LiteLLM_DailyCustomerMemberSpend" ADD COLUMN IF NOT EXISTS "app_id" TEXT NOT NULL DEFAULT '';
