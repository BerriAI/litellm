-- Add batch_processed column to LiteLLM_ManagedObjectTable
-- Set to true by CheckBatchCost after cost has been computed for a completed batch
ALTER TABLE "LiteLLM_ManagedObjectTable" ADD COLUMN "batch_processed" BOOLEAN NOT NULL DEFAULT false;
