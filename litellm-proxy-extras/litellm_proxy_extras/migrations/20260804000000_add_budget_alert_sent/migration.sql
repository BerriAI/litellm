-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_BudgetAlertSent" (
    "id" TEXT NOT NULL,
    "token" TEXT NOT NULL,
    "alert_type" TEXT NOT NULL,
    "threshold_pct" INTEGER NOT NULL,
    "budget_window" TEXT NOT NULL,
    "sent_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_BudgetAlertSent_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_BudgetAlertSent_token_alert_type_threshold_pct_key" ON "LiteLLM_BudgetAlertSent"("token", "alert_type", "threshold_pct");

-- AddForeignKey
ALTER TABLE "LiteLLM_BudgetAlertSent" DROP CONSTRAINT IF EXISTS "LiteLLM_BudgetAlertSent_token_fkey";
ALTER TABLE "LiteLLM_BudgetAlertSent" ADD CONSTRAINT "LiteLLM_BudgetAlertSent_token_fkey" FOREIGN KEY ("token") REFERENCES "LiteLLM_VerificationToken"("token") ON DELETE CASCADE ON UPDATE CASCADE;
