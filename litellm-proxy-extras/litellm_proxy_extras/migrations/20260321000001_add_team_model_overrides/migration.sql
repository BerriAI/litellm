-- AlterTable: Add default_models to LiteLLM_TeamTable
ALTER TABLE "LiteLLM_TeamTable" ADD COLUMN IF NOT EXISTS "default_models" TEXT[] DEFAULT ARRAY[]::TEXT[];

-- AlterTable: Add models to LiteLLM_TeamMembership
ALTER TABLE "LiteLLM_TeamMembership" ADD COLUMN IF NOT EXISTS "models" TEXT[] DEFAULT ARRAY[]::TEXT[];
