-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_BudgetAlertSent" (
    "id" TEXT NOT NULL,
    "entity_type" TEXT NOT NULL,
    "entity_id" TEXT NOT NULL,
    "alert_type" TEXT NOT NULL,
    "threshold_pct" INTEGER NOT NULL,
    "budget_window" TEXT NOT NULL,
    "sent_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_BudgetAlertSent_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_BudgetAlertSent_entity_type_entity_id_alert_type_th_key" ON "LiteLLM_BudgetAlertSent"("entity_type", "entity_id", "alert_type", "threshold_pct");
