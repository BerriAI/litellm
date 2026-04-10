-- AlterTable
ALTER TABLE "LiteLLM_PromptTable" ADD COLUMN "environment" TEXT NOT NULL DEFAULT 'development';
ALTER TABLE "LiteLLM_PromptTable" ADD COLUMN "created_by" TEXT;

-- DropIndex (old unique constraint)
DROP INDEX IF EXISTS "LiteLLM_PromptTable_prompt_id_version_key";

-- CreateIndex (new unique constraint)
CREATE UNIQUE INDEX "LiteLLM_PromptTable_prompt_id_version_environment_key" ON "LiteLLM_PromptTable"("prompt_id", "version", "environment");

-- CreateIndex (new composite index)
CREATE INDEX "LiteLLM_PromptTable_prompt_id_environment_idx" ON "LiteLLM_PromptTable"("prompt_id", "environment");
