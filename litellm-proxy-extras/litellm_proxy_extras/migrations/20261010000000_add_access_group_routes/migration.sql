-- AlterTable
ALTER TABLE "LiteLLM_AccessGroupTable" ADD COLUMN IF NOT EXISTS "access_routes" TEXT[] DEFAULT ARRAY[]::TEXT[];
