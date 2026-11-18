SET LOCAL lock_timeout = '10s';

CREATE TABLE IF NOT EXISTS "LiteLLM_CustomerOrganizationTable" (
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "budget_reset_at" TIMESTAMP(3),
    "customer_org_id" TEXT NOT NULL,
    "alias" TEXT,
    "default_budget_id" TEXT,
    "budget_ids" TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    "budget_names" JSONB NOT NULL DEFAULT '{}',
    "budget_id" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT NOT NULL,

    CONSTRAINT "LiteLLM_CustomerOrganizationTable_pkey" PRIMARY KEY ("customer_org_id")
);

CREATE TABLE IF NOT EXISTS "LiteLLM_CustomerTeamTable" (
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "budget_reset_at" TIMESTAMP(3),
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

CREATE TABLE IF NOT EXISTS "LiteLLM_CustomerMemberTable" (
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "budget_reset_at" TIMESTAMP(3),
    "customer_member_id" TEXT NOT NULL,
    "alias" TEXT,
    "parent_team_id" TEXT NOT NULL,
    "budget_id" TEXT,
    "budget_template_id" TEXT,
    "budget_override_expires_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT NOT NULL,

    CONSTRAINT "LiteLLM_CustomerMemberTable_pkey" PRIMARY KEY ("customer_member_id")
);

CREATE INDEX IF NOT EXISTS "LiteLLM_CustomerTeamTable_parent_org_id_idx" ON "LiteLLM_CustomerTeamTable"("parent_org_id");

CREATE INDEX IF NOT EXISTS "LiteLLM_CustomerMemberTable_parent_team_id_idx" ON "LiteLLM_CustomerMemberTable"("parent_team_id");

CREATE INDEX IF NOT EXISTS "LiteLLM_CustomerMemberTable_budget_template_id_idx" ON "LiteLLM_CustomerMemberTable"("budget_template_id");

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'LiteLLM_CustomerOrganizationTable_default_budget_id_fkey'
          AND conrelid = '"LiteLLM_CustomerOrganizationTable"'::regclass
    ) THEN
        ALTER TABLE "LiteLLM_CustomerOrganizationTable" ADD CONSTRAINT "LiteLLM_CustomerOrganizationTable_default_budget_id_fkey" FOREIGN KEY ("default_budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'LiteLLM_CustomerTeamTable_parent_org_id_fkey'
          AND conrelid = '"LiteLLM_CustomerTeamTable"'::regclass
    ) THEN
        ALTER TABLE "LiteLLM_CustomerTeamTable" ADD CONSTRAINT "LiteLLM_CustomerTeamTable_parent_org_id_fkey" FOREIGN KEY ("parent_org_id") REFERENCES "LiteLLM_CustomerOrganizationTable"("customer_org_id") ON DELETE RESTRICT ON UPDATE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'LiteLLM_CustomerTeamTable_budget_id_fkey'
          AND conrelid = '"LiteLLM_CustomerTeamTable"'::regclass
    ) THEN
        ALTER TABLE "LiteLLM_CustomerTeamTable" ADD CONSTRAINT "LiteLLM_CustomerTeamTable_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'LiteLLM_CustomerMemberTable_parent_team_id_fkey'
          AND conrelid = '"LiteLLM_CustomerMemberTable"'::regclass
    ) THEN
        ALTER TABLE "LiteLLM_CustomerMemberTable" ADD CONSTRAINT "LiteLLM_CustomerMemberTable_parent_team_id_fkey" FOREIGN KEY ("parent_team_id") REFERENCES "LiteLLM_CustomerTeamTable"("customer_team_id") ON DELETE RESTRICT ON UPDATE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'LiteLLM_CustomerMemberTable_budget_id_fkey'
          AND conrelid = '"LiteLLM_CustomerMemberTable"'::regclass
    ) THEN
        ALTER TABLE "LiteLLM_CustomerMemberTable" ADD CONSTRAINT "LiteLLM_CustomerMemberTable_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE RESTRICT ON UPDATE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'LiteLLM_CustomerOrganizationTable_budget_id_fkey'
          AND conrelid = '"LiteLLM_CustomerOrganizationTable"'::regclass
    ) THEN
        ALTER TABLE "LiteLLM_CustomerOrganizationTable" ADD CONSTRAINT "LiteLLM_CustomerOrganizationTable_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'LiteLLM_CustomerMemberTable_budget_template_id_fkey'
          AND conrelid = '"LiteLLM_CustomerMemberTable"'::regclass
    ) THEN
        ALTER TABLE "LiteLLM_CustomerMemberTable" ADD CONSTRAINT "LiteLLM_CustomerMemberTable_budget_template_id_fkey" FOREIGN KEY ("budget_template_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE RESTRICT ON UPDATE CASCADE;
    END IF;
END $$;
