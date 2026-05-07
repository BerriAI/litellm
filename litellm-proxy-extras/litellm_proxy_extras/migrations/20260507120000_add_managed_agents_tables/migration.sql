-- CreateTable
CREATE TABLE "LiteLLM_ManagedAgentSandboxTemplateTable" (
    "template_id" TEXT NOT NULL,
    "template_name" TEXT,
    "harness" TEXT NOT NULL,
    "image_uri" TEXT NOT NULL,
    "container_port" INTEGER NOT NULL DEFAULT 4096,
    "image_env" JSONB DEFAULT '{}',
    "repo_url" TEXT NOT NULL,
    "default_branch" TEXT NOT NULL DEFAULT 'main',
    "visibility" TEXT NOT NULL DEFAULT 'public',
    "git_credential_id" TEXT,
    "description" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL,
    "updated_by" TEXT,

    CONSTRAINT "LiteLLM_ManagedAgentSandboxTemplateTable_pkey" PRIMARY KEY ("template_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_ManagedAgentTable" (
    "agent_id" TEXT NOT NULL,
    "agent_name" TEXT,
    "model" TEXT NOT NULL,
    "prompt" TEXT,
    "tools" JSONB NOT NULL DEFAULT '[]',
    "template_id" TEXT NOT NULL,
    "branch" TEXT,
    "metadata" JSONB DEFAULT '{}',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "team_id" TEXT,
    "organization_id" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL,
    "updated_by" TEXT,

    CONSTRAINT "LiteLLM_ManagedAgentTable_pkey" PRIMARY KEY ("agent_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_ManagedAgentSessionTable" (
    "session_id" TEXT NOT NULL,
    "agent_id" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'creating',
    "task_arn" TEXT,
    "sandbox_url" TEXT,
    "harness_session_id" TEXT,
    "fargate_cluster" TEXT,
    "fargate_task_def_arn" TEXT,
    "virtual_key_hash" TEXT,
    "failure_reason" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "team_id" TEXT,
    "last_seen_at" TIMESTAMP(3),
    "expires_at" TIMESTAMP(3),
    "stopped_at" TIMESTAMP(3),

    CONSTRAINT "LiteLLM_ManagedAgentSessionTable_pkey" PRIMARY KEY ("session_id")
);

-- CreateIndex
CREATE INDEX "LiteLLM_ManagedAgentSandboxTemplateTable_harness_idx" ON "LiteLLM_ManagedAgentSandboxTemplateTable"("harness");

-- CreateIndex
CREATE INDEX "LiteLLM_ManagedAgentSandboxTemplateTable_created_by_idx" ON "LiteLLM_ManagedAgentSandboxTemplateTable"("created_by");

-- CreateIndex
CREATE INDEX "LiteLLM_ManagedAgentSandboxTemplateTable_git_credential_id_idx" ON "LiteLLM_ManagedAgentSandboxTemplateTable"("git_credential_id");

-- CreateIndex
CREATE INDEX "LiteLLM_ManagedAgentTable_template_id_idx" ON "LiteLLM_ManagedAgentTable"("template_id");

-- CreateIndex
CREATE INDEX "LiteLLM_ManagedAgentTable_created_by_idx" ON "LiteLLM_ManagedAgentTable"("created_by");

-- CreateIndex
CREATE INDEX "LiteLLM_ManagedAgentTable_team_id_idx" ON "LiteLLM_ManagedAgentTable"("team_id");

-- CreateIndex
CREATE INDEX "LiteLLM_ManagedAgentTable_organization_id_idx" ON "LiteLLM_ManagedAgentTable"("organization_id");

-- CreateIndex
CREATE INDEX "LiteLLM_ManagedAgentSessionTable_agent_id_idx" ON "LiteLLM_ManagedAgentSessionTable"("agent_id");

-- CreateIndex
CREATE INDEX "LiteLLM_ManagedAgentSessionTable_status_idx" ON "LiteLLM_ManagedAgentSessionTable"("status");

-- CreateIndex
CREATE INDEX "LiteLLM_ManagedAgentSessionTable_created_by_idx" ON "LiteLLM_ManagedAgentSessionTable"("created_by");

-- CreateIndex
CREATE INDEX "LiteLLM_ManagedAgentSessionTable_last_seen_at_idx" ON "LiteLLM_ManagedAgentSessionTable"("last_seen_at");

-- AddForeignKey
ALTER TABLE "LiteLLM_ManagedAgentSandboxTemplateTable" ADD CONSTRAINT "LiteLLM_ManagedAgentSandboxTemplateTable_git_credential_id_fkey" FOREIGN KEY ("git_credential_id") REFERENCES "LiteLLM_CredentialsTable"("credential_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_ManagedAgentTable" ADD CONSTRAINT "LiteLLM_ManagedAgentTable_template_id_fkey" FOREIGN KEY ("template_id") REFERENCES "LiteLLM_ManagedAgentSandboxTemplateTable"("template_id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_ManagedAgentSessionTable" ADD CONSTRAINT "LiteLLM_ManagedAgentSessionTable_agent_id_fkey" FOREIGN KEY ("agent_id") REFERENCES "LiteLLM_ManagedAgentTable"("agent_id") ON DELETE CASCADE ON UPDATE CASCADE;
