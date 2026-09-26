ALTER TABLE "LiteLLM_AgentIdentity" ADD COLUMN IF NOT EXISTS "provisioning_source_id" TEXT;

ALTER TABLE "LiteLLM_VerifiedSubject" ADD COLUMN IF NOT EXISTS "agent_id" TEXT,
ADD COLUMN IF NOT EXISTS "parent_client_id" TEXT,
ADD COLUMN IF NOT EXISTS "scim_resource_id" TEXT;

-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_SCIMSource" (
    "source_id" TEXT NOT NULL,
    "display_name" TEXT NOT NULL,
    "tenant_id" TEXT NOT NULL,
    "key_hash" TEXT NOT NULL,
    "enabled" BOOLEAN NOT NULL DEFAULT true,
    "group_mappings" JSONB NOT NULL DEFAULT '[]',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_SCIMSource_pkey" PRIMARY KEY ("source_id")
);

-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_SCIMResource" (
    "id" TEXT NOT NULL,
    "source_id" TEXT NOT NULL,
    "kind" TEXT NOT NULL,
    "external_id" TEXT NOT NULL,
    "user_name" TEXT,
    "display_name" TEXT NOT NULL,
    "document" JSONB NOT NULL,
    "active" BOOLEAN NOT NULL DEFAULT true,
    "deleted" BOOLEAN NOT NULL DEFAULT false,
    "local_id" TEXT,
    "human_email" TEXT,
    "human_subject_key" TEXT,
    "member_ids" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_SCIMResource_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_VerifiedSubject_scim_resource_id_key" ON "LiteLLM_VerifiedSubject"("scim_resource_id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_VerifiedSubject_agent_id_idx" ON "LiteLLM_VerifiedSubject"("agent_id");

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_SCIMSource_key_hash_key" ON "LiteLLM_SCIMSource"("key_hash");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_SCIMResource_source_id_kind_idx" ON "LiteLLM_SCIMResource"("source_id", "kind");

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_SCIMResource_source_id_kind_external_id_key" ON "LiteLLM_SCIMResource"("source_id", "kind", "external_id");

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_SCIMResource_source_id_kind_user_name_key" ON "LiteLLM_SCIMResource"("source_id", "kind", "user_name");

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_SCIMResource_local_id_key" ON "LiteLLM_SCIMResource"("local_id");

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_SCIMResource_human_email_key" ON "LiteLLM_SCIMResource"("human_email");

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_SCIMResource_human_subject_key_key" ON "LiteLLM_SCIMResource"("human_subject_key");

-- AddForeignKey
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_VerifiedSubject_agent_id_fkey') THEN
        ALTER TABLE "LiteLLM_VerifiedSubject" ADD CONSTRAINT "LiteLLM_VerifiedSubject_agent_id_fkey" FOREIGN KEY ("agent_id") REFERENCES "LiteLLM_AgentsTable"("agent_id") ON DELETE SET NULL ON UPDATE CASCADE;
    END IF;
END $$;

-- AddForeignKey
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_VerifiedSubject_scim_resource_id_fkey') THEN
        ALTER TABLE "LiteLLM_VerifiedSubject" ADD CONSTRAINT "LiteLLM_VerifiedSubject_scim_resource_id_fkey" FOREIGN KEY ("scim_resource_id") REFERENCES "LiteLLM_SCIMResource"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
    END IF;
END $$;

-- AddForeignKey
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_SCIMResource_source_id_fkey') THEN
        ALTER TABLE "LiteLLM_SCIMResource" ADD CONSTRAINT "LiteLLM_SCIMResource_source_id_fkey" FOREIGN KEY ("source_id") REFERENCES "LiteLLM_SCIMSource"("source_id") ON DELETE RESTRICT ON UPDATE CASCADE;
    END IF;
END $$;

-- AddCheckConstraint
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_VerifiedSubject_kind_shape') THEN
        ALTER TABLE "LiteLLM_VerifiedSubject" ADD CONSTRAINT "LiteLLM_VerifiedSubject_kind_shape" CHECK (
            ("kind" = 'human' AND "user_id" IS NOT NULL AND "agent_id" IS NULL
                AND "parent_client_id" IS NULL AND "scim_resource_id" IS NULL AND "verified_via" = 'sso_interactive')
            OR
            ("kind" = 'agent_user' AND "user_id" IS NULL AND "parent_client_id" IS NOT NULL
                AND "scim_resource_id" IS NOT NULL AND "verified_via" = 'scim')
        );
    END IF;
END $$;
