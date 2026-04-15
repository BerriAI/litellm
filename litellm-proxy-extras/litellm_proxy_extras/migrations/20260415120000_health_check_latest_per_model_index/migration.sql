-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_HealthCheckTable_model_id_model_name_checked_at_idx" ON "LiteLLM_HealthCheckTable"("model_id", "model_name", "checked_at" DESC);
