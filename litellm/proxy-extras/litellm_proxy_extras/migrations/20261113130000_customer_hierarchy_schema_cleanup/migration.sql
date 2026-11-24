SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN IF NOT EXISTS "name" TEXT;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'LiteLLM_CustomerOrganizationTable' AND column_name = 'budget_ids') THEN
        ALTER TABLE "LiteLLM_CustomerOrganizationTable" RENAME COLUMN "budget_ids" TO "budget_template_ids";
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'LiteLLM_CustomerOrganizationTable' AND column_name = 'default_budget_id') THEN
        ALTER TABLE "LiteLLM_CustomerOrganizationTable" RENAME COLUMN "default_budget_id" TO "default_budget_template_id";
    END IF;
    IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_CustomerOrganizationTable_default_budget_id_fkey') THEN
        ALTER TABLE "LiteLLM_CustomerOrganizationTable" RENAME CONSTRAINT "LiteLLM_CustomerOrganizationTable_default_budget_id_fkey" TO "LiteLLM_CustomerOrganizationTable_default_budget_template_id_fkey";
    END IF;
END $$;
ALTER TABLE "LiteLLM_CustomerOrganizationTable" DROP COLUMN IF EXISTS "budget_names";

ALTER TABLE "LiteLLM_CustomerTeamTable" ADD COLUMN IF NOT EXISTS "budget_override_id" TEXT;
ALTER TABLE "LiteLLM_CustomerTeamTable" ADD COLUMN IF NOT EXISTS "budget_override_expires_at" TIMESTAMP(3);
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_CustomerTeamTable_budget_override_id_fkey') THEN
        ALTER TABLE "LiteLLM_CustomerTeamTable" ADD CONSTRAINT "LiteLLM_CustomerTeamTable_budget_override_id_fkey" FOREIGN KEY ("budget_override_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE RESTRICT ON UPDATE CASCADE;
    END IF;
END $$;

ALTER TABLE "LiteLLM_CustomerMemberTable" ADD COLUMN IF NOT EXISTS "parent_org_id" TEXT;
ALTER TABLE "LiteLLM_CustomerMemberTable" ADD COLUMN IF NOT EXISTS "parent_team_ids" TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[];
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'LiteLLM_CustomerMemberTable' AND column_name = 'parent_team_id') THEN
        -- data-migration-ok: customer tables exist only on the v1.111.0 dev line and hold a handful of rows
        UPDATE "LiteLLM_CustomerMemberTable" m SET "parent_org_id" = t."parent_org_id", "parent_team_ids" = ARRAY[m."parent_team_id"] FROM "LiteLLM_CustomerTeamTable" t WHERE m."parent_team_id" = t."customer_team_id" AND m."parent_org_id" IS NULL;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_CustomerMemberTable_parent_team_id_fkey') THEN
        ALTER TABLE "LiteLLM_CustomerMemberTable" DROP CONSTRAINT "LiteLLM_CustomerMemberTable_parent_team_id_fkey";
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_CustomerMemberTable_parent_org_id_fkey') THEN
        ALTER TABLE "LiteLLM_CustomerMemberTable" ADD CONSTRAINT "LiteLLM_CustomerMemberTable_parent_org_id_fkey" FOREIGN KEY ("parent_org_id") REFERENCES "LiteLLM_CustomerOrganizationTable"("customer_org_id") ON DELETE RESTRICT ON UPDATE CASCADE;
    END IF;
END $$;
ALTER TABLE "LiteLLM_CustomerMemberTable" ALTER COLUMN "parent_org_id" SET NOT NULL;
DROP INDEX IF EXISTS "LiteLLM_CustomerMemberTable_parent_team_id_idx";
DROP INDEX IF EXISTS "LiteLLM_CustomerMemberTable_budget_id_idx";
ALTER TABLE "LiteLLM_CustomerMemberTable" DROP COLUMN IF EXISTS "parent_team_id";

CREATE INDEX IF NOT EXISTS "LiteLLM_CustomerMemberTable_parent_org_id_idx" ON "LiteLLM_CustomerMemberTable"("parent_org_id");
CREATE INDEX IF NOT EXISTS "LiteLLM_CustomerOrganizationTable_spend_idx" ON "LiteLLM_CustomerOrganizationTable"("spend");
CREATE INDEX IF NOT EXISTS "LiteLLM_CustomerTeamTable_spend_idx" ON "LiteLLM_CustomerTeamTable"("spend");
CREATE INDEX IF NOT EXISTS "LiteLLM_CustomerMemberTable_spend_idx" ON "LiteLLM_CustomerMemberTable"("spend");
