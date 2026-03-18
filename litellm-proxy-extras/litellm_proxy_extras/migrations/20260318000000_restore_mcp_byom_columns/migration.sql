-- Migration: Restore BYOM columns that were incorrectly dropped by 20260311180521_schema_sync.
-- schema.prisma still defines all six fields; re-adding them here keeps the DB in sync.

-- Restore source_url
ALTER TABLE "LiteLLM_MCPServerTable"
  ADD COLUMN IF NOT EXISTS "source_url"      TEXT,
  ADD COLUMN IF NOT EXISTS "approval_status" TEXT DEFAULT 'active',
  ADD COLUMN IF NOT EXISTS "submitted_by"    TEXT,
  ADD COLUMN IF NOT EXISTS "submitted_at"    TIMESTAMP(3),
  ADD COLUMN IF NOT EXISTS "reviewed_at"     TIMESTAMP(3),
  ADD COLUMN IF NOT EXISTS "review_notes"    TEXT;

-- Re-create the index dropped by 20260311180521_schema_sync
CREATE INDEX IF NOT EXISTS "LiteLLM_MCPServerTable_approval_status_idx"
  ON "LiteLLM_MCPServerTable"("approval_status");
