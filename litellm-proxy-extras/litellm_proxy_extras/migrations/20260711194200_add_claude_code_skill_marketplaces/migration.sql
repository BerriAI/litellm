-- AlterTable
ALTER TABLE "LiteLLM_ClaudeCodePluginTable" ADD COLUMN     "marketplace_id" TEXT;

-- AlterTable
ALTER TABLE "LiteLLM_ObjectPermissionTable" ADD COLUMN     "allowed_skills" TEXT[] DEFAULT ARRAY[]::TEXT[];

-- CreateTable
CREATE TABLE "LiteLLM_SkillMarketplaceTable" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "display_name" TEXT,
    "source_type" TEXT NOT NULL,
    "source_ref" TEXT,
    "branch" TEXT DEFAULT 'main',
    "owner_name" TEXT,
    "owner_email" TEXT,
    "enabled" BOOLEAN NOT NULL DEFAULT true,
    "sync_status" TEXT NOT NULL DEFAULT 'pending',
    "sync_error" TEXT,
    "skipped_count" INTEGER NOT NULL DEFAULT 0,
    "last_synced_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,

    CONSTRAINT "LiteLLM_SkillMarketplaceTable_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_SkillMarketplaceTable_name_key" ON "LiteLLM_SkillMarketplaceTable"("name");

