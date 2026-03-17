-- DropIndex
DROP INDEX IF EXISTS "LiteLLM_MCPServerTable_approval_status_idx";

-- AlterTable
ALTER TABLE "LiteLLM_MCPServerTable" DROP COLUMN IF EXISTS "approval_status",
DROP COLUMN IF EXISTS "review_notes",
DROP COLUMN IF EXISTS "reviewed_at",
DROP COLUMN IF EXISTS "source_url",
DROP COLUMN IF EXISTS "submitted_at",
DROP COLUMN IF EXISTS "submitted_by";

