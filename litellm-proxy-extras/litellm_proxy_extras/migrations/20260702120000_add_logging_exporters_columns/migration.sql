-- AlterTable
ALTER TABLE "LiteLLM_TeamTable" ADD COLUMN IF NOT EXISTS "logging_exporters" TEXT[] DEFAULT ARRAY[]::TEXT[];

-- AlterTable
ALTER TABLE "LiteLLM_DeletedTeamTable" ADD COLUMN IF NOT EXISTS "logging_exporters" TEXT[] DEFAULT ARRAY[]::TEXT[];

-- AlterTable
ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN IF NOT EXISTS "logging_exporters" TEXT[] DEFAULT ARRAY[]::TEXT[];

-- AlterTable
ALTER TABLE "LiteLLM_DeletedVerificationToken" ADD COLUMN IF NOT EXISTS "logging_exporters" TEXT[] DEFAULT ARRAY[]::TEXT[];

-- AlterTable
ALTER TABLE "LiteLLM_OrganizationTable" ADD COLUMN IF NOT EXISTS "logging_exporters" TEXT[] DEFAULT ARRAY[]::TEXT[];
