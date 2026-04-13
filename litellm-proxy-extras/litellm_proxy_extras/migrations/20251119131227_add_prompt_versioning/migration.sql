-- DropIndex
DROP INDEX IF EXISTS "LiteLLM_PromptTable_prompt_id_key";

-- AlterTable
ALTER TABLE "LiteLLM_PromptTable"
ADD COLUMN IF NOT EXISTS "version" INTEGER NOT NULL DEFAULT 1;

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_PromptTable_prompt_id_idx" ON "LiteLLM_PromptTable" ("prompt_id");

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_PromptTable_prompt_id_version_key" ON "LiteLLM_PromptTable" ("prompt_id", "version");