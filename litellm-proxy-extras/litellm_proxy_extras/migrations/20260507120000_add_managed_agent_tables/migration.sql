-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_ManagedAgent" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "config" JSONB NOT NULL,
    "created_by" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_ManagedAgent_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_ManagedAgentSession" (
    "id" TEXT NOT NULL,
    "agent_id" TEXT NOT NULL,
    "sandbox_type" TEXT NOT NULL,
    "sandbox_size" TEXT NOT NULL,
    "sandbox_timeout_minutes" INTEGER NOT NULL,
    "sandbox_idle_timeout_minutes" INTEGER NOT NULL,
    "sandbox_image" TEXT,
    "sandbox_url" TEXT,
    "sandbox_metadata" JSONB,
    "status" TEXT NOT NULL DEFAULT 'provisioning',
    "repos" JSONB,
    "env_vars" JSONB,
    "created_by" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,
    "terminated_at" TIMESTAMP(3),

    CONSTRAINT "LiteLLM_ManagedAgentSession_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_ManagedAgentSession_agent_id_idx" ON "LiteLLM_ManagedAgentSession"("agent_id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_ManagedAgentSession_created_by_idx" ON "LiteLLM_ManagedAgentSession"("created_by");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_ManagedAgentSession_status_idx" ON "LiteLLM_ManagedAgentSession"("status");

-- CreateIndex
-- Backstop for the application-level (name, created_by) duplicate check —
-- catches the read-then-write race when two concurrent POST /v2/agents
-- requests both pass the pre-query and try to insert at the same time.
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_ManagedAgent_name_created_by_key" ON "LiteLLM_ManagedAgent"("name", "created_by");
