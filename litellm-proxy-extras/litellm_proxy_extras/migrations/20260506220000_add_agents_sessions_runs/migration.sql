-- CreateTable
CREATE TABLE "LiteLLM_Agent" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "user_api_key_hash" TEXT NOT NULL,
    "team_id" TEXT,
    "model" TEXT NOT NULL,
    "system_prompt" TEXT,
    "default_repos" JSONB,
    "default_env_vars" JSONB,
    "tools_config" JSONB,
    "metadata" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_Agent_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_AgentSession" (
    "id" TEXT NOT NULL,
    "agent_id" TEXT NOT NULL,
    "user_api_key_hash" TEXT NOT NULL,
    "team_id" TEXT,
    "vm_id" TEXT,
    "vm_provider" TEXT,
    "repos" JSONB NOT NULL,
    "env_vars" JSONB,
    "status" TEXT NOT NULL DEFAULT 'provisioning',
    "daemon_token_hash" TEXT,
    "expires_at" TIMESTAMP(3) NOT NULL,
    "last_heartbeat_at" TIMESTAMP(3),
    "idempotency_key" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,
    "terminated_at" TIMESTAMP(3),

    CONSTRAINT "LiteLLM_AgentSession_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_AgentRun" (
    "id" TEXT NOT NULL,
    "session_id" TEXT NOT NULL,
    "parent_run_id" TEXT,
    "status" TEXT NOT NULL DEFAULT 'queued',
    "prompt" JSONB NOT NULL,
    "result" TEXT,
    "git_branches" JSONB,
    "idempotency_key" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,
    "started_at" TIMESTAMP(3),
    "terminated_at" TIMESTAMP(3),

    CONSTRAINT "LiteLLM_AgentRun_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_AgentRunEvent" (
    "id" TEXT NOT NULL,
    "run_id" TEXT NOT NULL,
    "seq" INTEGER NOT NULL,
    "event_type" TEXT NOT NULL,
    "payload" JSONB NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_AgentRunEvent_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "LiteLLM_Agent_user_api_key_hash_idx" ON "LiteLLM_Agent"("user_api_key_hash");

-- CreateIndex
CREATE INDEX "LiteLLM_Agent_team_id_idx" ON "LiteLLM_Agent"("team_id");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_AgentSession_user_api_key_hash_idempotency_key_key" ON "LiteLLM_AgentSession"("user_api_key_hash", "idempotency_key");

-- CreateIndex
CREATE INDEX "LiteLLM_AgentSession_agent_id_idx" ON "LiteLLM_AgentSession"("agent_id");

-- CreateIndex
CREATE INDEX "LiteLLM_AgentSession_status_expires_at_idx" ON "LiteLLM_AgentSession"("status", "expires_at");

-- CreateIndex
CREATE INDEX "LiteLLM_AgentSession_user_api_key_hash_idx" ON "LiteLLM_AgentSession"("user_api_key_hash");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_AgentRun_session_id_idempotency_key_key" ON "LiteLLM_AgentRun"("session_id", "idempotency_key");

-- CreateIndex
CREATE INDEX "LiteLLM_AgentRun_session_id_status_idx" ON "LiteLLM_AgentRun"("session_id", "status");

-- CreateIndex
CREATE INDEX "LiteLLM_AgentRun_session_id_created_at_idx" ON "LiteLLM_AgentRun"("session_id", "created_at");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_AgentRunEvent_run_id_seq_key" ON "LiteLLM_AgentRunEvent"("run_id", "seq");

-- CreateIndex
CREATE INDEX "LiteLLM_AgentRunEvent_run_id_seq_idx" ON "LiteLLM_AgentRunEvent"("run_id", "seq");

-- AddForeignKey
ALTER TABLE "LiteLLM_AgentSession" ADD CONSTRAINT "LiteLLM_AgentSession_agent_id_fkey" FOREIGN KEY ("agent_id") REFERENCES "LiteLLM_Agent"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_AgentRun" ADD CONSTRAINT "LiteLLM_AgentRun_session_id_fkey" FOREIGN KEY ("session_id") REFERENCES "LiteLLM_AgentSession"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_AgentRunEvent" ADD CONSTRAINT "LiteLLM_AgentRunEvent_run_id_fkey" FOREIGN KEY ("run_id") REFERENCES "LiteLLM_AgentRun"("id") ON DELETE CASCADE ON UPDATE CASCADE;
