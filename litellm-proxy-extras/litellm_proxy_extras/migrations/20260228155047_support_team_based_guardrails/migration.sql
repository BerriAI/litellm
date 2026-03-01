-- AlterTable
ALTER TABLE "LiteLLM_GuardrailsTable" ADD COLUMN     "reviewed_at" TIMESTAMP(3),
ADD COLUMN     "status" TEXT NOT NULL DEFAULT 'active',
ADD COLUMN     "submitted_at" TIMESTAMP(3),
ADD COLUMN     "submitted_by_email" TEXT,
ADD COLUMN     "submitted_by_user_id" TEXT;

-- CreateIndex
CREATE INDEX "LiteLLM_GuardrailsTable_status_idx" ON "LiteLLM_GuardrailsTable"("status");

