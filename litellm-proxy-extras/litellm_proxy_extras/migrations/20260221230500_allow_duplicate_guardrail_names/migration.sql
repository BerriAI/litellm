-- Drop unique constraint on guardrail_name and add non-unique index for lookup
DROP INDEX IF EXISTS "LiteLLM_GuardrailsTable_guardrail_name_key";

CREATE INDEX IF NOT EXISTS "LiteLLM_GuardrailsTable_guardrail_name_idx"
ON "LiteLLM_GuardrailsTable"("guardrail_name");
