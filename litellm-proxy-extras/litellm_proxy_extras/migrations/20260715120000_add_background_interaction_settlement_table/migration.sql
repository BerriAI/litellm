-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_BackgroundInteractionSettlementTable" (
    "interaction_id" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "outcome" TEXT,
    "context" JSONB NOT NULL,
    "claimed_by" TEXT,
    "timeout_at" TIMESTAMP(3) NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_BackgroundInteractionSettlementTable_pkey" PRIMARY KEY ("interaction_id")
);

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_BackgroundInteractionSettlement_status_created_idx" ON "LiteLLM_BackgroundInteractionSettlementTable"("status", "created_at");
