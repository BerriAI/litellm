-- Schema v2 for managed-agents sandbox template:
-- replace `harness` with `dockerfile_id`, drop `image_env`, make `image_uri` nullable,
-- and add build-pipeline fields (task_def_arn, image_hash, build_status, build_error).

-- DropIndex
DROP INDEX "LiteLLM_ManagedAgentSandboxTemplateTable_harness_idx";

-- AlterTable
ALTER TABLE "LiteLLM_ManagedAgentSandboxTemplateTable"
    DROP COLUMN "harness",
    DROP COLUMN "image_env",
    ALTER COLUMN "image_uri" DROP NOT NULL,
    ADD COLUMN "task_def_arn" TEXT,
    ADD COLUMN "image_hash" TEXT,
    ADD COLUMN "build_status" TEXT NOT NULL DEFAULT 'pending',
    ADD COLUMN "build_error" TEXT;

-- Add NOT NULL `dockerfile_id`. We use a transient default of '' so the ADD COLUMN
-- succeeds even on a populated table, then drop the default so future inserts must
-- supply a real value. There is no production data on this table yet (v1 just shipped),
-- so no backfill statement is required.
ALTER TABLE "LiteLLM_ManagedAgentSandboxTemplateTable"
    ADD COLUMN "dockerfile_id" TEXT NOT NULL DEFAULT '';
ALTER TABLE "LiteLLM_ManagedAgentSandboxTemplateTable"
    ALTER COLUMN "dockerfile_id" DROP DEFAULT;

-- CreateIndex
CREATE INDEX "LiteLLM_ManagedAgentSandboxTemplateTable_dockerfile_id_idx" ON "LiteLLM_ManagedAgentSandboxTemplateTable"("dockerfile_id");
