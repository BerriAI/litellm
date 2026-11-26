SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_DailyCustomerMemberSpend" ALTER COLUMN "updated_at" DROP DEFAULT;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'LiteLLM_CustomerOrganizationTable_default_budget_template_id_fk'
          AND conrelid = '"LiteLLM_CustomerOrganizationTable"'::regclass
    ) THEN
        ALTER TABLE "LiteLLM_CustomerOrganizationTable" RENAME CONSTRAINT "LiteLLM_CustomerOrganizationTable_default_budget_template_id_fk" TO "LiteLLM_CustomerOrganizationTable_default_budget_template__fkey";
    END IF;
END $$;
