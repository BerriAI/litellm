SET LOCAL lock_timeout = '10s';

-- AlterTable
ALTER TABLE "LiteLLM_AppTable" ADD COLUMN     "app_group_id" TEXT;

-- AlterTable
ALTER TABLE "LiteLLM_DailyCustomerMemberSpend" ADD COLUMN     "app_group_id" TEXT NOT NULL DEFAULT '';

-- CreateTable
CREATE TABLE "LiteLLM_AppGroupTable" (
    "app_group_id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "metadata" JSONB,
    "budget_id" TEXT,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "spend_reset_window" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT NOT NULL,

    CONSTRAINT "LiteLLM_AppGroupTable_pkey" PRIMARY KEY ("app_group_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_AppGroupCustomerOrgTable" (
    "app_group_id" TEXT NOT NULL,
    "customer_org_id" TEXT NOT NULL,
    "budget_id" TEXT,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "spend_reset_window" TEXT,

    CONSTRAINT "LiteLLM_AppGroupCustomerOrgTable_pkey" PRIMARY KEY ("app_group_id","customer_org_id")
);

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_AppGroupTable_name_key" ON "LiteLLM_AppGroupTable"("name");

-- CreateIndex
CREATE INDEX "LiteLLM_AppGroupCustomerOrgTable_customer_org_id_idx" ON "LiteLLM_AppGroupCustomerOrgTable"("customer_org_id");

-- CreateIndex
CREATE INDEX "LiteLLM_AppTable_app_group_id_idx" ON "LiteLLM_AppTable"("app_group_id");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyCustomerMemberSpend_app_group_id_customer_org__idx" ON "LiteLLM_DailyCustomerMemberSpend"("app_group_id", "customer_org_id", "date");

-- AddForeignKey
ALTER TABLE "LiteLLM_AppGroupTable" ADD CONSTRAINT "LiteLLM_AppGroupTable_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_AppGroupCustomerOrgTable" ADD CONSTRAINT "LiteLLM_AppGroupCustomerOrgTable_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_AppGroupCustomerOrgTable" ADD CONSTRAINT "LiteLLM_AppGroupCustomerOrgTable_app_group_id_fkey" FOREIGN KEY ("app_group_id") REFERENCES "LiteLLM_AppGroupTable"("app_group_id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_AppGroupCustomerOrgTable" ADD CONSTRAINT "LiteLLM_AppGroupCustomerOrgTable_customer_org_id_fkey" FOREIGN KEY ("customer_org_id") REFERENCES "LiteLLM_CustomerOrganizationTable"("customer_org_id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_AppTable" ADD CONSTRAINT "LiteLLM_AppTable_app_group_id_fkey" FOREIGN KEY ("app_group_id") REFERENCES "LiteLLM_AppGroupTable"("app_group_id") ON DELETE RESTRICT ON UPDATE CASCADE;

