ALTER TABLE "LiteLLM_BudgetTable"
  ADD COLUMN IF NOT EXISTS "temp_budget_duration" TEXT,
  ADD COLUMN IF NOT EXISTS "budget_limits" JSONB,
  ADD COLUMN IF NOT EXISTS "is_profile" BOOLEAN NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS "cap_policy" TEXT;

ALTER TABLE "LiteLLM_UserTable"
  ADD COLUMN IF NOT EXISTS "budget_template_id" TEXT,
  ADD COLUMN IF NOT EXISTS "budget_id" TEXT,
  ADD COLUMN IF NOT EXISTS "budget_override_expires_at" TIMESTAMP(3);

CREATE INDEX IF NOT EXISTS "LiteLLM_UserTable_budget_template_id_idx" ON "LiteLLM_UserTable"("budget_template_id");
CREATE INDEX IF NOT EXISTS "LiteLLM_UserTable_budget_id_idx" ON "LiteLLM_UserTable"("budget_id");

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_UserTable_budget_template_id_fkey') THEN
    ALTER TABLE "LiteLLM_UserTable" ADD CONSTRAINT "LiteLLM_UserTable_budget_template_id_fkey"
      FOREIGN KEY ("budget_template_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_UserTable_budget_id_fkey') THEN
    ALTER TABLE "LiteLLM_UserTable" ADD CONSTRAINT "LiteLLM_UserTable_budget_id_fkey"
      FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE;
  END IF;
END $$;
