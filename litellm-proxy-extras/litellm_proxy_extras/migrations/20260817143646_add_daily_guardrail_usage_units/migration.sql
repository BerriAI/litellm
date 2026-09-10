-- CreateTable
CREATE TABLE "LiteLLM_DailyGuardrailUsageUnits" (
    "guardrail_id" TEXT NOT NULL,
    "date" TEXT NOT NULL,
    "team_id" TEXT NOT NULL,
    "api_key" TEXT NOT NULL,
    "usage_unit" TEXT NOT NULL,
    "units" BIGINT NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_DailyGuardrailUsageUnits_pkey" PRIMARY KEY ("guardrail_id","date","team_id","api_key","usage_unit")
);

-- CreateIndex
CREATE INDEX "LiteLLM_DailyGuardrailUsageUnits_date_idx" ON "LiteLLM_DailyGuardrailUsageUnits"("date");
