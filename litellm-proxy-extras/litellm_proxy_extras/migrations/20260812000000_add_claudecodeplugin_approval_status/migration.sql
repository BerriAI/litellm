-- AlterTable
ALTER TABLE "LiteLLM_ClaudeCodePluginTable" ADD COLUMN IF NOT EXISTS "approval_status" TEXT DEFAULT 'active';
ALTER TABLE "LiteLLM_ClaudeCodePluginTable" ADD COLUMN IF NOT EXISTS "review_notes" TEXT;
ALTER TABLE "LiteLLM_ClaudeCodePluginTable" ADD COLUMN IF NOT EXISTS "reviewed_by" TEXT;
ALTER TABLE "LiteLLM_ClaudeCodePluginTable" ADD COLUMN IF NOT EXISTS "reviewed_at" TIMESTAMP(3);

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_ClaudeCodePluginTable_approval_status_idx" ON "LiteLLM_ClaudeCodePluginTable"("approval_status");
