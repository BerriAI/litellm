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
