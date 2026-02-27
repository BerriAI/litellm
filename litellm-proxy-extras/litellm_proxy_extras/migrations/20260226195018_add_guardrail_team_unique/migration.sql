-- DropIndex
DROP INDEX "LiteLLM_GuardrailsTable_guardrail_name_key";

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_GuardrailsTable_guardrail_name_team_id_key" ON "LiteLLM_GuardrailsTable"("guardrail_name", "team_id");

