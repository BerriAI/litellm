CREATE TABLE "LiteLLM_CustomerOrganizationTable" (
    "customer_org_id" TEXT NOT NULL,
    "alias" TEXT,
    "default_budget_id" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT NOT NULL,

    CONSTRAINT "LiteLLM_CustomerOrganizationTable_pkey" PRIMARY KEY ("customer_org_id")
);

CREATE TABLE "LiteLLM_CustomerOrganizationBudgetTable" (
    "customer_org_id" TEXT NOT NULL,
    "budget_id" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_CustomerOrganizationBudgetTable_pkey" PRIMARY KEY ("customer_org_id","budget_id")
);

CREATE TABLE "LiteLLM_CustomerTeamTable" (
    "customer_team_id" TEXT NOT NULL,
    "alias" TEXT,
    "parent_org_id" TEXT NOT NULL,
    "budget_id" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT NOT NULL,

    CONSTRAINT "LiteLLM_CustomerTeamTable_pkey" PRIMARY KEY ("customer_team_id")
);

CREATE TABLE "LiteLLM_CustomerMemberTable" (
    "customer_member_id" TEXT NOT NULL,
    "alias" TEXT,
    "parent_team_id" TEXT NOT NULL,
    "budget_id" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT NOT NULL,

    CONSTRAINT "LiteLLM_CustomerMemberTable_pkey" PRIMARY KEY ("customer_member_id")
);

CREATE UNIQUE INDEX "LiteLLM_CustomerOrganizationBudgetTable_budget_id_key" ON "LiteLLM_CustomerOrganizationBudgetTable"("budget_id");

CREATE INDEX "LiteLLM_CustomerTeamTable_parent_org_id_idx" ON "LiteLLM_CustomerTeamTable"("parent_org_id");

CREATE INDEX "LiteLLM_CustomerMemberTable_parent_team_id_idx" ON "LiteLLM_CustomerMemberTable"("parent_team_id");

ALTER TABLE "LiteLLM_CustomerOrganizationTable" ADD CONSTRAINT "LiteLLM_CustomerOrganizationTable_customer_org_id_default__fkey" FOREIGN KEY ("customer_org_id", "default_budget_id") REFERENCES "LiteLLM_CustomerOrganizationBudgetTable"("customer_org_id", "budget_id") ON DELETE NO ACTION ON UPDATE NO ACTION;

ALTER TABLE "LiteLLM_CustomerOrganizationBudgetTable" ADD CONSTRAINT "LiteLLM_CustomerOrganizationBudgetTable_customer_org_id_fkey" FOREIGN KEY ("customer_org_id") REFERENCES "LiteLLM_CustomerOrganizationTable"("customer_org_id") ON DELETE RESTRICT ON UPDATE CASCADE;

ALTER TABLE "LiteLLM_CustomerOrganizationBudgetTable" ADD CONSTRAINT "LiteLLM_CustomerOrganizationBudgetTable_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE RESTRICT ON UPDATE CASCADE;

ALTER TABLE "LiteLLM_CustomerTeamTable" ADD CONSTRAINT "LiteLLM_CustomerTeamTable_parent_org_id_fkey" FOREIGN KEY ("parent_org_id") REFERENCES "LiteLLM_CustomerOrganizationTable"("customer_org_id") ON DELETE RESTRICT ON UPDATE CASCADE;

ALTER TABLE "LiteLLM_CustomerTeamTable" ADD CONSTRAINT "LiteLLM_CustomerTeamTable_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE;

ALTER TABLE "LiteLLM_CustomerMemberTable" ADD CONSTRAINT "LiteLLM_CustomerMemberTable_parent_team_id_fkey" FOREIGN KEY ("parent_team_id") REFERENCES "LiteLLM_CustomerTeamTable"("customer_team_id") ON DELETE RESTRICT ON UPDATE CASCADE;

ALTER TABLE "LiteLLM_CustomerMemberTable" ADD CONSTRAINT "LiteLLM_CustomerMemberTable_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE RESTRICT ON UPDATE CASCADE;
