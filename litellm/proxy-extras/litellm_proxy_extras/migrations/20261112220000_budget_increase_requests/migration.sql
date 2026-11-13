SET LOCAL lock_timeout = '10s';

CREATE TABLE IF NOT EXISTS "LiteLLM_BudgetIncreaseRequest" (
    "id" TEXT NOT NULL,
    "submitted_by" TEXT NOT NULL,
    "pending_user_id" TEXT,
    "amount" DOUBLE PRECISION NOT NULL,
    "duration" TEXT NOT NULL,
    "expires_at" TIMESTAMPTZ(6) NOT NULL,
    "reason" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'pending_review',
    "created_at" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "reviewed_by" TEXT,
    "reviewed_at" TIMESTAMPTZ(6),
    "rejection_reason" TEXT,
    CONSTRAINT "LiteLLM_BudgetIncreaseRequest_pkey" PRIMARY KEY ("id")
);
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_BudgetIncreaseRequest_pending_user_id_key" ON "LiteLLM_BudgetIncreaseRequest"("pending_user_id");
CREATE INDEX IF NOT EXISTS "LiteLLM_BudgetIncreaseRequest_submitted_by_created_at_id_idx" ON "LiteLLM_BudgetIncreaseRequest"("submitted_by", "created_at", "id");
CREATE INDEX IF NOT EXISTS "LiteLLM_BudgetIncreaseRequest_created_at_id_idx" ON "LiteLLM_BudgetIncreaseRequest"("created_at", "id");
