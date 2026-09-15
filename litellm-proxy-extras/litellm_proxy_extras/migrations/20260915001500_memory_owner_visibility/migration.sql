ALTER TABLE "LiteLLM_MemoryTable" ADD COLUMN IF NOT EXISTS "organization_id" TEXT;
ALTER TABLE "LiteLLM_MemoryTable" ADD COLUMN IF NOT EXISTS "owner_key_id" TEXT;
CREATE INDEX IF NOT EXISTS "LiteLLM_MemoryTable_organization_id_user_id_updated_at_idx" ON "LiteLLM_MemoryTable"("organization_id", "user_id", "updated_at");
CREATE INDEX IF NOT EXISTS "LiteLLM_MemoryTable_team_id_updated_at_idx" ON "LiteLLM_MemoryTable"("team_id", "updated_at");
CREATE INDEX IF NOT EXISTS "LiteLLM_MemoryTable_owner_key_id_idx" ON "LiteLLM_MemoryTable"("owner_key_id");
