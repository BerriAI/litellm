ALTER TABLE "LiteLLM_EndUserTable" ADD COLUMN IF NOT EXISTS "fallback_end_user_id" TEXT;
CREATE INDEX IF NOT EXISTS "LiteLLM_EndUserTable_fallback_end_user_id_idx" ON "LiteLLM_EndUserTable"("fallback_end_user_id");
