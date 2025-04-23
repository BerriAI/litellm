-- CreateTable
CREATE TABLE "LiteLLM_BudgetTable" (
    "budget_id" TEXT NOT NULL,
    "max_budget" DOUBLE PRECISION,
    "soft_budget" DOUBLE PRECISION,
    "max_parallel_requests" INTEGER,
    "tpm_limit" BIGINT,
    "rpm_limit" BIGINT,
    "model_max_budget" JSONB,
    "budget_duration" TEXT,
    "budget_reset_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT NOT NULL,

    CONSTRAINT "LiteLLM_BudgetTable_pkey" PRIMARY KEY ("budget_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_CredentialsTable" (
    "credential_id" TEXT NOT NULL,
    "credential_name" TEXT NOT NULL,
    "credential_values" JSONB NOT NULL,
    "credential_info" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT NOT NULL,

    CONSTRAINT "LiteLLM_CredentialsTable_pkey" PRIMARY KEY ("credential_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_ProxyModelTable" (
    "model_id" TEXT NOT NULL,
    "model_name" TEXT NOT NULL,
    "litellm_params" JSONB NOT NULL,
    "model_info" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT NOT NULL,

    CONSTRAINT "LiteLLM_ProxyModelTable_pkey" PRIMARY KEY ("model_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_OrganizationTable" (
    "organization_id" TEXT NOT NULL,
    "organization_alias" TEXT NOT NULL,
    "budget_id" TEXT NOT NULL,
    "metadata" JSONB NOT NULL DEFAULT '{}',
    "models" TEXT[],
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "model_spend" JSONB NOT NULL DEFAULT '{}',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT NOT NULL,

    CONSTRAINT "LiteLLM_OrganizationTable_pkey" PRIMARY KEY ("organization_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_ModelTable" (
    "id" SERIAL NOT NULL,
    "aliases" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT NOT NULL,

    CONSTRAINT "LiteLLM_ModelTable_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_TeamTable" (
    "team_id" TEXT NOT NULL,
    "team_alias" TEXT,
    "organization_id" TEXT,
    "admins" TEXT[],
    "members" TEXT[],
    "members_with_roles" JSONB NOT NULL DEFAULT '{}',
    "metadata" JSONB NOT NULL DEFAULT '{}',
    "max_budget" DOUBLE PRECISION,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "models" TEXT[],
    "max_parallel_requests" INTEGER,
    "tpm_limit" BIGINT,
    "rpm_limit" BIGINT,
    "budget_duration" TEXT,
    "budget_reset_at" TIMESTAMP(3),
    "blocked" BOOLEAN NOT NULL DEFAULT false,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "model_spend" JSONB NOT NULL DEFAULT '{}',
    "model_max_budget" JSONB NOT NULL DEFAULT '{}',
    "model_id" INTEGER,

    CONSTRAINT "LiteLLM_TeamTable_pkey" PRIMARY KEY ("team_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_UserTable" (
    "user_id" TEXT NOT NULL,
    "user_alias" TEXT,
    "team_id" TEXT,
    "sso_user_id" TEXT,
    "organization_id" TEXT,
    "password" TEXT,
    "teams" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "user_role" TEXT,
    "max_budget" DOUBLE PRECISION,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "user_email" TEXT,
    "models" TEXT[],
    "metadata" JSONB NOT NULL DEFAULT '{}',
    "max_parallel_requests" INTEGER,
    "tpm_limit" BIGINT,
    "rpm_limit" BIGINT,
    "budget_duration" TEXT,
    "budget_reset_at" TIMESTAMP(3),
    "allowed_cache_controls" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "model_spend" JSONB NOT NULL DEFAULT '{}',
    "model_max_budget" JSONB NOT NULL DEFAULT '{}',
    "created_at" TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_UserTable_pkey" PRIMARY KEY ("user_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_VerificationToken" (
    "token" TEXT NOT NULL,
    "key_name" TEXT,
    "key_alias" TEXT,
    "soft_budget_cooldown" BOOLEAN NOT NULL DEFAULT false,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "expires" TIMESTAMP(3),
    "models" TEXT[],
    "aliases" JSONB NOT NULL DEFAULT '{}',
    "config" JSONB NOT NULL DEFAULT '{}',
    "user_id" TEXT,
    "team_id" TEXT,
    "permissions" JSONB NOT NULL DEFAULT '{}',
    "max_parallel_requests" INTEGER,
    "metadata" JSONB NOT NULL DEFAULT '{}',
    "blocked" BOOLEAN,
    "tpm_limit" BIGINT,
    "rpm_limit" BIGINT,
    "max_budget" DOUBLE PRECISION,
    "budget_duration" TEXT,
    "budget_reset_at" TIMESTAMP(3),
    "allowed_cache_controls" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "model_spend" JSONB NOT NULL DEFAULT '{}',
    "model_max_budget" JSONB NOT NULL DEFAULT '{}',
    "budget_id" TEXT,
    "organization_id" TEXT,
    "created_at" TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT,

    CONSTRAINT "LiteLLM_VerificationToken_pkey" PRIMARY KEY ("token")
);

-- CreateTable
CREATE TABLE "LiteLLM_EndUserTable" (
    "user_id" TEXT NOT NULL,
    "alias" TEXT,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "allowed_model_region" TEXT,
    "default_model" TEXT,
    "budget_id" TEXT,
    "blocked" BOOLEAN NOT NULL DEFAULT false,

    CONSTRAINT "LiteLLM_EndUserTable_pkey" PRIMARY KEY ("user_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_Config" (
    "param_name" TEXT NOT NULL,
    "param_value" JSONB,

    CONSTRAINT "LiteLLM_Config_pkey" PRIMARY KEY ("param_name")
);

-- CreateTable
CREATE TABLE "LiteLLM_SpendLogs" (
    "request_id" TEXT NOT NULL,
    "call_type" TEXT NOT NULL,
    "api_key" TEXT NOT NULL DEFAULT '',
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "total_tokens" INTEGER NOT NULL DEFAULT 0,
    "prompt_tokens" INTEGER NOT NULL DEFAULT 0,
    "completion_tokens" INTEGER NOT NULL DEFAULT 0,
    "startTime" TIMESTAMP(3) NOT NULL,
    "endTime" TIMESTAMP(3) NOT NULL,
    "completionStartTime" TIMESTAMP(3),
    "model" TEXT NOT NULL DEFAULT '',
    "model_id" TEXT DEFAULT '',
    "model_group" TEXT DEFAULT '',
    "custom_llm_provider" TEXT DEFAULT '',
    "api_base" TEXT DEFAULT '',
    "user" TEXT DEFAULT '',
    "metadata" JSONB DEFAULT '{}',
    "cache_hit" TEXT DEFAULT '',
    "cache_key" TEXT DEFAULT '',
    "request_tags" JSONB DEFAULT '[]',
    "team_id" TEXT,
    "end_user" TEXT,
    "requester_ip_address" TEXT,
    "messages" JSONB DEFAULT '{}',
    "response" JSONB DEFAULT '{}',

    CONSTRAINT "LiteLLM_SpendLogs_pkey" PRIMARY KEY ("request_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_ErrorLogs" (
    "request_id" TEXT NOT NULL,
    "startTime" TIMESTAMP(3) NOT NULL,
    "endTime" TIMESTAMP(3) NOT NULL,
    "api_base" TEXT NOT NULL DEFAULT '',
    "model_group" TEXT NOT NULL DEFAULT '',
    "litellm_model_name" TEXT NOT NULL DEFAULT '',
    "model_id" TEXT NOT NULL DEFAULT '',
    "request_kwargs" JSONB NOT NULL DEFAULT '{}',
    "exception_type" TEXT NOT NULL DEFAULT '',
    "exception_string" TEXT NOT NULL DEFAULT '',
    "status_code" TEXT NOT NULL DEFAULT '',

    CONSTRAINT "LiteLLM_ErrorLogs_pkey" PRIMARY KEY ("request_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_UserNotifications" (
    "request_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "models" TEXT[],
    "justification" TEXT NOT NULL,
    "status" TEXT NOT NULL,

    CONSTRAINT "LiteLLM_UserNotifications_pkey" PRIMARY KEY ("request_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_TeamMembership" (
    "user_id" TEXT NOT NULL,
    "team_id" TEXT NOT NULL,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "budget_id" TEXT,

    CONSTRAINT "LiteLLM_TeamMembership_pkey" PRIMARY KEY ("user_id","team_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_OrganizationMembership" (
    "user_id" TEXT NOT NULL,
    "organization_id" TEXT NOT NULL,
    "user_role" TEXT,
    "spend" DOUBLE PRECISION DEFAULT 0.0,
    "budget_id" TEXT,
    "created_at" TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_OrganizationMembership_pkey" PRIMARY KEY ("user_id","organization_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_InvitationLink" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "is_accepted" BOOLEAN NOT NULL DEFAULT false,
    "accepted_at" TIMESTAMP(3),
    "expires_at" TIMESTAMP(3) NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL,
    "created_by" TEXT NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL,
    "updated_by" TEXT NOT NULL,

    CONSTRAINT "LiteLLM_InvitationLink_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_AuditLog" (
    "id" TEXT NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "changed_by" TEXT NOT NULL DEFAULT '',
    "changed_by_api_key" TEXT NOT NULL DEFAULT '',
    "action" TEXT NOT NULL,
    "table_name" TEXT NOT NULL,
    "object_id" TEXT NOT NULL,
    "before_value" JSONB,
    "updated_values" JSONB,

    CONSTRAINT "LiteLLM_AuditLog_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_CredentialsTable_credential_name_key" ON "LiteLLM_CredentialsTable"("credential_name");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_TeamTable_model_id_key" ON "LiteLLM_TeamTable"("model_id");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_UserTable_sso_user_id_key" ON "LiteLLM_UserTable"("sso_user_id");

-- CreateIndex
CREATE INDEX "LiteLLM_SpendLogs_startTime_idx" ON "LiteLLM_SpendLogs"("startTime");

-- CreateIndex
CREATE INDEX "LiteLLM_SpendLogs_end_user_idx" ON "LiteLLM_SpendLogs"("end_user");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_OrganizationMembership_user_id_organization_id_key" ON "LiteLLM_OrganizationMembership"("user_id", "organization_id");

-- AddForeignKey
ALTER TABLE "LiteLLM_OrganizationTable" ADD CONSTRAINT "LiteLLM_OrganizationTable_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_TeamTable" ADD CONSTRAINT "LiteLLM_TeamTable_organization_id_fkey" FOREIGN KEY ("organization_id") REFERENCES "LiteLLM_OrganizationTable"("organization_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_TeamTable" ADD CONSTRAINT "LiteLLM_TeamTable_model_id_fkey" FOREIGN KEY ("model_id") REFERENCES "LiteLLM_ModelTable"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_UserTable" ADD CONSTRAINT "LiteLLM_UserTable_organization_id_fkey" FOREIGN KEY ("organization_id") REFERENCES "LiteLLM_OrganizationTable"("organization_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_VerificationToken" ADD CONSTRAINT "LiteLLM_VerificationToken_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_VerificationToken" ADD CONSTRAINT "LiteLLM_VerificationToken_organization_id_fkey" FOREIGN KEY ("organization_id") REFERENCES "LiteLLM_OrganizationTable"("organization_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_EndUserTable" ADD CONSTRAINT "LiteLLM_EndUserTable_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_TeamMembership" ADD CONSTRAINT "LiteLLM_TeamMembership_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_OrganizationMembership" ADD CONSTRAINT "LiteLLM_OrganizationMembership_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "LiteLLM_UserTable"("user_id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_OrganizationMembership" ADD CONSTRAINT "LiteLLM_OrganizationMembership_organization_id_fkey" FOREIGN KEY ("organization_id") REFERENCES "LiteLLM_OrganizationTable"("organization_id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_OrganizationMembership" ADD CONSTRAINT "LiteLLM_OrganizationMembership_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_InvitationLink" ADD CONSTRAINT "LiteLLM_InvitationLink_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "LiteLLM_UserTable"("user_id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_InvitationLink" ADD CONSTRAINT "LiteLLM_InvitationLink_created_by_fkey" FOREIGN KEY ("created_by") REFERENCES "LiteLLM_UserTable"("user_id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_InvitationLink" ADD CONSTRAINT "LiteLLM_InvitationLink_updated_by_fkey" FOREIGN KEY ("updated_by") REFERENCES "LiteLLM_UserTable"("user_id") ON DELETE RESTRICT ON UPDATE CASCADE;

