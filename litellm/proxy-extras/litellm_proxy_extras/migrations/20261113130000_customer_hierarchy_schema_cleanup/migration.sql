SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN IF NOT EXISTS "name" TEXT;

ALTER TABLE "LiteLLM_CustomerOrganizationTable" RENAME COLUMN "budget_ids" TO "budget_template_ids";
ALTER TABLE "LiteLLM_CustomerOrganizationTable" RENAME COLUMN "default_budget_id" TO "default_budget_template_id";
ALTER TABLE "LiteLLM_CustomerOrganizationTable" RENAME CONSTRAINT "LiteLLM_CustomerOrganizationTable_default_budget_id_fkey" TO "LiteLLM_CustomerOrganizationTable_default_budget_template_id_fkey";
ALTER TABLE "LiteLLM_CustomerOrganizationTable" DROP COLUMN IF EXISTS "budget_names";

ALTER TABLE "LiteLLM_CustomerTeamTable" ADD COLUMN IF NOT EXISTS "budget_override_id" TEXT;
ALTER TABLE "LiteLLM_CustomerTeamTable" ADD COLUMN IF NOT EXISTS "budget_override_expires_at" TIMESTAMP(3);
ALTER TABLE "LiteLLM_CustomerTeamTable" ADD CONSTRAINT "LiteLLM_CustomerTeamTable_budget_override_id_fkey" FOREIGN KEY ("budget_override_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE RESTRICT ON UPDATE CASCADE;

ALTER TABLE "LiteLLM_CustomerMemberTable" ADD COLUMN IF NOT EXISTS "parent_org_id" TEXT;
-- data-migration-ok: customer tables exist only on the v1.111.0 dev line and hold a handful of rows
UPDATE "LiteLLM_CustomerMemberTable" m SET "parent_org_id" = t."parent_org_id" FROM "LiteLLM_CustomerTeamTable" t WHERE m."parent_team_id" = t."customer_team_id" AND m."parent_org_id" IS NULL;
ALTER TABLE "LiteLLM_CustomerMemberTable" ALTER COLUMN "parent_org_id" SET NOT NULL;
ALTER TABLE "LiteLLM_CustomerMemberTable" ADD CONSTRAINT "LiteLLM_CustomerMemberTable_parent_org_id_fkey" FOREIGN KEY ("parent_org_id") REFERENCES "LiteLLM_CustomerOrganizationTable"("customer_org_id") ON DELETE RESTRICT ON UPDATE CASCADE;
CREATE INDEX IF NOT EXISTS "LiteLLM_CustomerMemberTable_parent_org_id_idx" ON "LiteLLM_CustomerMemberTable"("parent_org_id");
