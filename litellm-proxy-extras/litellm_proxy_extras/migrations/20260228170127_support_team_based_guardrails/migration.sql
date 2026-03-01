-- AlterTable
ALTER TABLE "LiteLLM_GuardrailsTable" ADD COLUMN     "reviewed_at" TIMESTAMP(3),
ADD COLUMN     "status" TEXT NOT NULL DEFAULT 'active',
ADD COLUMN     "submitted_at" TIMESTAMP(3);

-- CreateIndex
CREATE INDEX "LiteLLM_GuardrailsTable_status_idx" ON "LiteLLM_GuardrailsTable"("status");

