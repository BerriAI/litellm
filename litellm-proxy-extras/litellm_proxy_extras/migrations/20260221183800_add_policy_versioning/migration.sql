-- DropIndex
DROP INDEX IF EXISTS "LiteLLM_PolicyTable_policy_name_key";

-- AlterTable
ALTER TABLE "LiteLLM_PolicyTable" ADD COLUMN IF NOT EXISTS "is_latest" BOOLEAN NOT NULL DEFAULT true,
ADD COLUMN IF NOT EXISTS "parent_version_id" TEXT,
ADD COLUMN IF NOT EXISTS "production_at" TIMESTAMP(3),
ADD COLUMN IF NOT EXISTS "published_at" TIMESTAMP(3),
ADD COLUMN IF NOT EXISTS "version_number" INTEGER NOT NULL DEFAULT 1,
ADD COLUMN IF NOT EXISTS "version_status" TEXT NOT NULL DEFAULT 'production';

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_PolicyTable_policy_name_version_status_idx" ON "LiteLLM_PolicyTable"("policy_name", "version_status");

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_PolicyTable_policy_name_version_number_key" ON "LiteLLM_PolicyTable"("policy_name", "version_number");

