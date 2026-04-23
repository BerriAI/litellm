-- AlterTable
ALTER TABLE "LiteLLM_ManagedFileTable" ADD COLUMN IF NOT EXISTS "storage_backend" TEXT;
ALTER TABLE "LiteLLM_ManagedFileTable" ADD COLUMN IF NOT EXISTS "storage_url" TEXT;

