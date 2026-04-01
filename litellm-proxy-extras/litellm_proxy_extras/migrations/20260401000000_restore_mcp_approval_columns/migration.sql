-- Restore BYOM approval workflow columns that were accidentally dropped by
-- 20260311180521_schema_sync.  That migration was auto-generated because the
-- root schema.prisma was not updated when PR #23205 added the submission
-- workflow.  This migration re-adds the columns and index.

ALTER TABLE "LiteLLM_MCPServerTable"
  ADD COLUMN IF NOT EXISTS "source_url"       TEXT,
  ADD COLUMN IF NOT EXISTS "approval_status"  TEXT DEFAULT 'active',
  ADD COLUMN IF NOT EXISTS "submitted_by"     TEXT,
  ADD COLUMN IF NOT EXISTS "submitted_at"     TIMESTAMP(3),
  ADD COLUMN IF NOT EXISTS "reviewed_at"      TIMESTAMP(3),
  ADD COLUMN IF NOT EXISTS "review_notes"     TEXT;

-- Back-fill existing rows: anything already in the table is implicitly active.
-- Also normalise the old "approved" default written by a prior schema version
-- that used @default("approved") instead of @default("active").
UPDATE "LiteLLM_MCPServerTable"
  SET "approval_status" = 'active'
  WHERE "approval_status" IS NULL OR "approval_status" = 'approved';

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_MCPServerTable_approval_status_idx"
  ON "LiteLLM_MCPServerTable"("approval_status");
