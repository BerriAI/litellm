-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_SubscriptionAccountTable" (
    "subscription_account_id" TEXT NOT NULL,
    "custom_llm_provider" TEXT NOT NULL,
    "account_id" TEXT NOT NULL,
    "label" TEXT,
    "monthly_fee" DOUBLE PRECISION NOT NULL,
    "currency" TEXT NOT NULL,
    "billing_period_start" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT NOT NULL,

    CONSTRAINT "LiteLLM_SubscriptionAccountTable_pkey" PRIMARY KEY ("subscription_account_id")
);

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_SubscriptionAccountTable_provider_account_key" ON "LiteLLM_SubscriptionAccountTable"("custom_llm_provider", "account_id");
