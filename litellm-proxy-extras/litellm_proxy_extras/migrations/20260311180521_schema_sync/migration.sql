-- DropIndex
DROP INDEX "LiteLLM_MCPServerTable_approval_status_idx";

-- AlterTable
ALTER TABLE "LiteLLM_MCPServerTable" DROP COLUMN "approval_status",
DROP COLUMN "review_notes",
DROP COLUMN "reviewed_at",
DROP COLUMN "source_url",
DROP COLUMN "submitted_at",
DROP COLUMN "submitted_by";

