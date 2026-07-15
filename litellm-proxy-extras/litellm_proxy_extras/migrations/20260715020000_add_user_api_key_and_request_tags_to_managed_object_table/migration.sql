-- Add user_api_key and request_tags columns to LiteLLM_ManagedObjectTable
-- Captured at batch-create time so CheckBatchCost can attribute batch-cost spend
-- back to the creating virtual key (and its tags) even when created_by is null.
ALTER TABLE "LiteLLM_ManagedObjectTable" ADD COLUMN IF NOT EXISTS "user_api_key" TEXT;
ALTER TABLE "LiteLLM_ManagedObjectTable" ADD COLUMN IF NOT EXISTS "request_tags" JSONB;
