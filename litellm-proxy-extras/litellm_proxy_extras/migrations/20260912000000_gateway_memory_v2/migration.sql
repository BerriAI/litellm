ALTER TABLE "LiteLLM_MemoryTable" ADD COLUMN IF NOT EXISTS "namespace" TEXT;

CREATE INDEX IF NOT EXISTS "LiteLLM_MemoryTable_namespace_updated_at_idx"
ON "LiteLLM_MemoryTable"("namespace", "updated_at");

ALTER TABLE "LiteLLM_MemoryTable" ADD COLUMN IF NOT EXISTS "organization_id" TEXT;
ALTER TABLE "LiteLLM_MemoryTable" ADD COLUMN IF NOT EXISTS "owner_key_id" TEXT;
CREATE INDEX IF NOT EXISTS "LiteLLM_MemoryTable_organization_id_user_id_updated_at_idx" ON "LiteLLM_MemoryTable"("organization_id", "user_id", "updated_at");
CREATE INDEX IF NOT EXISTS "LiteLLM_MemoryTable_team_id_updated_at_idx" ON "LiteLLM_MemoryTable"("team_id", "updated_at");
CREATE INDEX IF NOT EXISTS "LiteLLM_MemoryTable_owner_key_id_idx" ON "LiteLLM_MemoryTable"("owner_key_id");

CREATE TABLE IF NOT EXISTS "LiteLLM_MemoryContinuation" (
    "id" TEXT NOT NULL,
    "namespace" TEXT NOT NULL,
    "key_id" TEXT NOT NULL,
    "payload" JSONB NOT NULL,
    "expires_at" TIMESTAMP(3) NOT NULL,
    CONSTRAINT "LiteLLM_MemoryContinuation_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "LiteLLM_MemoryContinuation_namespace_key_id_expires_at_idx"
ON "LiteLLM_MemoryContinuation"("namespace", "key_id", "expires_at");

CREATE INDEX IF NOT EXISTS "LiteLLM_MemoryContinuation_expires_at_idx"
ON "LiteLLM_MemoryContinuation"("expires_at");
