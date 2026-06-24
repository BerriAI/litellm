-- AlterTable: add columns to LiteLLM_ServiceAccountTable
ALTER TABLE "LiteLLM_ServiceAccountTable" ADD COLUMN IF NOT EXISTS "name" TEXT;
ALTER TABLE "LiteLLM_ServiceAccountTable" ADD COLUMN IF NOT EXISTS "requested_models" TEXT[] DEFAULT ARRAY[]::TEXT[];
ALTER TABLE "LiteLLM_ServiceAccountTable" ADD COLUMN IF NOT EXISTS "use_case" TEXT;
ALTER TABLE "LiteLLM_ServiceAccountTable" ADD COLUMN IF NOT EXISTS "requested_rpm_limit" INTEGER;
ALTER TABLE "LiteLLM_ServiceAccountTable" ADD COLUMN IF NOT EXISTS "requested_parallel_requests_limit" INTEGER;
ALTER TABLE "LiteLLM_ServiceAccountTable" ADD COLUMN IF NOT EXISTS "is_active" BOOLEAN DEFAULT false;
ALTER TABLE "LiteLLM_ServiceAccountTable" ADD COLUMN IF NOT EXISTS "is_key_rotation_requested" BOOLEAN DEFAULT false;
