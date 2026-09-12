ALTER TABLE "LiteLLM_MemoryTable" ADD COLUMN IF NOT EXISTS "namespace" TEXT;

CREATE INDEX IF NOT EXISTS "LiteLLM_MemoryTable_namespace_updated_at_idx"
ON "LiteLLM_MemoryTable"("namespace", "updated_at");

CREATE TABLE IF NOT EXISTS "LiteLLM_MemoryPolicy" (
    "policy_id" TEXT NOT NULL,
    "target_type" TEXT NOT NULL,
    "target_id" TEXT NOT NULL,
    "activation" TEXT NOT NULL,
    "scope" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT NOT NULL,
    CONSTRAINT "LiteLLM_MemoryPolicy_pkey" PRIMARY KEY ("policy_id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_MemoryPolicy_target_type_target_id_key"
ON "LiteLLM_MemoryPolicy"("target_type", "target_id");

CREATE TABLE IF NOT EXISTS "LiteLLM_MemoryPreference" (
    "subject" TEXT NOT NULL,
    "enabled" BOOLEAN NOT NULL DEFAULT false,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "LiteLLM_MemoryPreference_pkey" PRIMARY KEY ("subject")
);
