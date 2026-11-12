SET LOCAL lock_timeout = '10s';

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_attribute
        WHERE attrelid = to_regclass('"LiteLLM_CustomerMemberTable"')
          AND attname = 'budget_id' AND NOT attisdropped
    ) THEN
        ALTER TABLE "LiteLLM_CustomerMemberTable" RENAME COLUMN "budget_id" TO "budget_override_id";
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'LiteLLM_CustomerMemberTable_budget_id_fkey'
          AND conrelid = to_regclass('"LiteLLM_CustomerMemberTable"')
    ) THEN
        ALTER TABLE "LiteLLM_CustomerMemberTable" RENAME CONSTRAINT "LiteLLM_CustomerMemberTable_budget_id_fkey" TO "LiteLLM_CustomerMemberTable_budget_override_id_fkey";
    END IF;
END $$;
