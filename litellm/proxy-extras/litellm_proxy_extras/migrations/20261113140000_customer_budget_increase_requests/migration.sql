SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN IF NOT EXISTS "cap_policy" TEXT;

CREATE TABLE IF NOT EXISTS "LiteLLM_CustomerBudgetIncreaseRequest" (
    "request_id" TEXT NOT NULL,
    "customer_member_id" TEXT NOT NULL,
    "requested_max_budget" DOUBLE PRECISION NOT NULL,
    "expires_at" TIMESTAMP(3),
    "reason" TEXT,
    "status" TEXT NOT NULL DEFAULT 'pending_review',
    "submitted_by" TEXT NOT NULL,
    "submitted_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "reviewed_by" TEXT,
    "reviewed_at" TIMESTAMP(3),
    "review_notes" TEXT,
    CONSTRAINT "LiteLLM_CustomerBudgetIncreaseRequest_pkey" PRIMARY KEY ("request_id")
);

CREATE INDEX IF NOT EXISTS "LiteLLM_CustomerBudgetIncreaseRequest_status_idx" ON "LiteLLM_CustomerBudgetIncreaseRequest"("status");
CREATE INDEX IF NOT EXISTS "LiteLLM_CustomerBudgetIncreaseRequest_customer_member_id_idx" ON "LiteLLM_CustomerBudgetIncreaseRequest"("customer_member_id");

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_CustomerBudgetIncreaseRequest_customer_member_id_fkey') THEN
        ALTER TABLE "LiteLLM_CustomerBudgetIncreaseRequest" ADD CONSTRAINT "LiteLLM_CustomerBudgetIncreaseRequest_customer_member_id_fkey" FOREIGN KEY ("customer_member_id") REFERENCES "LiteLLM_CustomerMemberTable"("customer_member_id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;
