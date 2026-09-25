-- AlterTable
ALTER TABLE "LiteLLM_AgentsTable" ADD COLUMN     "budget_id" TEXT,
ADD COLUMN     "enabled" BOOLEAN NOT NULL DEFAULT true,
ADD COLUMN     "execution_mode" TEXT NOT NULL DEFAULT 'autonomous',
ADD COLUMN     "identity_managed" BOOLEAN NOT NULL DEFAULT false;

-- AlterTable
ALTER TABLE "LiteLLM_SpendLogs" ADD COLUMN     "billing_agent_id" TEXT;

-- CreateTable
CREATE TABLE "LiteLLM_AgentIdentity" (
    "agent_id" TEXT NOT NULL,
    "active" BOOLEAN NOT NULL DEFAULT true,
    "provider" TEXT NOT NULL,
    "issuer" TEXT NOT NULL,
    "tenant_id" TEXT NOT NULL,
    "client_id" TEXT NOT NULL,
    "service_principal_id" TEXT,
    "required_roles" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "required_scopes" TEXT[] DEFAULT ARRAY['user_impersonation']::TEXT[],
    "revision" TEXT NOT NULL,
    "last_authenticated_at" TIMESTAMP(3),

    CONSTRAINT "LiteLLM_AgentIdentity_pkey" PRIMARY KEY ("agent_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_RetiredAgentIdentity" (
    "binding_id" TEXT NOT NULL,
    "agent_id" TEXT,
    "provider" TEXT NOT NULL,
    "issuer" TEXT NOT NULL,
    "tenant_id" TEXT NOT NULL,
    "client_id" TEXT NOT NULL,

    CONSTRAINT "LiteLLM_RetiredAgentIdentity_pkey" PRIMARY KEY ("binding_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_VerifiedHumanSubject" (
    "subject_id" TEXT NOT NULL,
    "issuer" TEXT NOT NULL,
    "tenant_id" TEXT NOT NULL,
    "oid" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "verified_via" TEXT NOT NULL DEFAULT 'sso_interactive',
    "verified_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_VerifiedHumanSubject_pkey" PRIMARY KEY ("subject_id")
);

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_AgentIdentity_provider_tenant_id_client_id_key" ON "LiteLLM_AgentIdentity"("provider", "tenant_id", "client_id");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_AgentIdentity_issuer_service_principal_id_key" ON "LiteLLM_AgentIdentity"("issuer", "service_principal_id");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_RetiredAgentIdentity_provider_tenant_id_client_id_key" ON "LiteLLM_RetiredAgentIdentity"("provider", "tenant_id", "client_id");

-- CreateIndex
CREATE INDEX "LiteLLM_VerifiedHumanSubject_user_id_idx" ON "LiteLLM_VerifiedHumanSubject"("user_id");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_VerifiedHumanSubject_issuer_tenant_id_oid_key" ON "LiteLLM_VerifiedHumanSubject"("issuer", "tenant_id", "oid");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_AgentsTable_budget_id_key" ON "LiteLLM_AgentsTable"("budget_id");

-- AddForeignKey
ALTER TABLE "LiteLLM_AgentsTable" ADD CONSTRAINT "LiteLLM_AgentsTable_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_AgentIdentity" ADD CONSTRAINT "LiteLLM_AgentIdentity_agent_id_fkey" FOREIGN KEY ("agent_id") REFERENCES "LiteLLM_AgentsTable"("agent_id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_RetiredAgentIdentity" ADD CONSTRAINT "LiteLLM_RetiredAgentIdentity_agent_id_fkey" FOREIGN KEY ("agent_id") REFERENCES "LiteLLM_AgentsTable"("agent_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_VerifiedHumanSubject" ADD CONSTRAINT "LiteLLM_VerifiedHumanSubject_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "LiteLLM_UserTable"("user_id") ON DELETE CASCADE ON UPDATE CASCADE;
