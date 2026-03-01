-- Migrate submitted_by_user_id and submitted_by_email into guardrail_info JSON, then drop columns
UPDATE "LiteLLM_GuardrailsTable"
SET guardrail_info = COALESCE(guardrail_info, '{}'::jsonb) || jsonb_build_object(
  'submitted_by_user_id', "submitted_by_user_id",
  'submitted_by_email', "submitted_by_email"
);

-- AlterTable
ALTER TABLE "LiteLLM_GuardrailsTable" DROP COLUMN "submitted_by_user_id";
ALTER TABLE "LiteLLM_GuardrailsTable" DROP COLUMN "submitted_by_email";
