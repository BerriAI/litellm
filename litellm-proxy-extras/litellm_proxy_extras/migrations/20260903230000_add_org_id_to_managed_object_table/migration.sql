-- Add org_id column to LiteLLM_ManagedObjectTable
-- Snapshots the creating key's organization at submission time, like team_id,
-- so CheckBatchCost can bill organization spend hours later without re-resolving
ALTER TABLE "LiteLLM_ManagedObjectTable" ADD COLUMN IF NOT EXISTS "org_id" TEXT;
