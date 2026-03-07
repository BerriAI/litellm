-- DropIndex
DROP INDEX "LiteLLM_PolicyTable_policy_name_key";

-- AlterTable
ALTER TABLE "LiteLLM_PolicyTable" ADD COLUMN     "is_latest" BOOLEAN NOT NULL DEFAULT true,
ADD COLUMN     "parent_version_id" TEXT,
ADD COLUMN     "production_at" TIMESTAMP(3),
ADD COLUMN     "published_at" TIMESTAMP(3),
ADD COLUMN     "version_number" INTEGER NOT NULL DEFAULT 1,
ADD COLUMN     "version_status" TEXT NOT NULL DEFAULT 'production';

-- CreateIndex
CREATE INDEX "LiteLLM_PolicyTable_policy_name_version_status_idx" ON "LiteLLM_PolicyTable"("policy_name", "version_status");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_PolicyTable_policy_name_version_number_key" ON "LiteLLM_PolicyTable"("policy_name", "version_number");

