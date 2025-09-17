-- Migration: Add is_active field to LiteLLM_ProxyModelTable
-- This migration adds the ability to enable/disable models without deleting them

-- Add is_active column to LiteLLM_ProxyModelTable if it doesn't exist
ALTER TABLE "LiteLLM_ProxyModelTable" 
ADD COLUMN IF NOT EXISTS "is_active" BOOLEAN NOT NULL DEFAULT true;

-- Create index on is_active for faster filtering
CREATE INDEX IF NOT EXISTS "idx_proxy_model_is_active" ON "LiteLLM_ProxyModelTable"("is_active");

-- Update any existing models to be active by default (if column was just added)
UPDATE "LiteLLM_ProxyModelTable" 
SET "is_active" = true 
WHERE "is_active" IS NULL;