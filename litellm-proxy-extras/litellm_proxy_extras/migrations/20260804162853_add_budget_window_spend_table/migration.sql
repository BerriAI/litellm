-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_BudgetWindowSpend" (
    "entity_type" TEXT NOT NULL,
    "entity_id" TEXT NOT NULL,
    "window_duration" TEXT NOT NULL,
    "window_start" TIMESTAMP(3) NOT NULL,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_BudgetWindowSpend_pkey" PRIMARY KEY ("entity_type","entity_id","window_duration")
);

