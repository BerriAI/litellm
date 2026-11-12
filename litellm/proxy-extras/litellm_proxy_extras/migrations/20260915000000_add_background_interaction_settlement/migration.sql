-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_BackgroundInteractionSettlement" (
    "interaction_id" TEXT NOT NULL,
    "custom_llm_provider" TEXT NOT NULL,
    "create_context" JSONB NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "claimed_at" TIMESTAMP(3),
    "claimed_by" TEXT,
    "settled_at" TIMESTAMP(3),
    "outcome" TEXT,

    CONSTRAINT "LiteLLM_BackgroundInteractionSettlement_pkey" PRIMARY KEY ("interaction_id")
);

-- CreateIndex
CREATE INDEX IF NOT EXISTS "idx_background_interaction_settlement_claimed_at" ON "LiteLLM_BackgroundInteractionSettlement"("claimed_at");
