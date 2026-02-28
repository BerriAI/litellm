-- AlterTable: add submission lifecycle columns to LiteLLM_GuardrailsTable
-- status: pending_review (team-registered), active (approved), rejected. Default active for existing rows.
ALTER TABLE "LiteLLM_GuardrailsTable" ADD COLUMN "status" TEXT NOT NULL DEFAULT 'active';
ALTER TABLE "LiteLLM_GuardrailsTable" ADD COLUMN "submitted_by_user_id" TEXT;
ALTER TABLE "LiteLLM_GuardrailsTable" ADD COLUMN "submitted_by_email" TEXT;
ALTER TABLE "LiteLLM_GuardrailsTable" ADD COLUMN "submitted_at" TIMESTAMP(3);
ALTER TABLE "LiteLLM_GuardrailsTable" ADD COLUMN "reviewed_at" TIMESTAMP(3);

-- CreateIndex
CREATE INDEX "LiteLLM_GuardrailsTable_status_idx" ON "LiteLLM_GuardrailsTable"("status");
