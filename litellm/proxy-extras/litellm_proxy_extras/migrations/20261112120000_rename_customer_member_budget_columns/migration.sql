SET LOCAL lock_timeout = '10s';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_attribute
        WHERE attrelid = to_regclass('"LiteLLM_CustomerMemberTable"')
          AND attname = 'budget_template_id' AND NOT attisdropped
    ) THEN
        RETURN;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_attribute
        WHERE attrelid = to_regclass('"LiteLLM_CustomerMemberTable"')
          AND attname = 'budget_override_id' AND NOT attisdropped
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
    ALTER TABLE "LiteLLM_CustomerMemberTable" RENAME COLUMN "budget_template_id" TO "budget_id";
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'LiteLLM_CustomerMemberTable_budget_template_id_fkey'
          AND conrelid = to_regclass('"LiteLLM_CustomerMemberTable"')
    ) THEN
        ALTER TABLE "LiteLLM_CustomerMemberTable" RENAME CONSTRAINT "LiteLLM_CustomerMemberTable_budget_template_id_fkey" TO "LiteLLM_CustomerMemberTable_budget_id_fkey";
    END IF;
END $$;

ALTER INDEX IF EXISTS "LiteLLM_CustomerMemberTable_budget_template_id_idx" RENAME TO "LiteLLM_CustomerMemberTable_budget_id_idx";
