-- CreateTable
CREATE TABLE "LiteLLM_GuardrailsTable" (
    "guardrail_id" TEXT NOT NULL,
    "guardrail_name" TEXT NOT NULL,
    "litellm_params" JSONB NOT NULL,
    "guardrail_info" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_GuardrailsTable_pkey" PRIMARY KEY ("guardrail_id")
);

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_GuardrailsTable_guardrail_name_key" ON "LiteLLM_GuardrailsTable"("guardrail_name");

