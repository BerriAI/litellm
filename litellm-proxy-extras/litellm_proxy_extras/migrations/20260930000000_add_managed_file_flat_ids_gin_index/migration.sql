-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_ManagedFileTable_flat_model_file_ids_idx" ON "LiteLLM_ManagedFileTable" USING GIN ("flat_model_file_ids");
