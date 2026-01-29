-- CreateEnum
CREATE TYPE "JobStatus" AS ENUM ('ACTIVE', 'INACTIVE');

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
CREATE TABLE "LiteLLM_AgentsTable" (
    "agent_id" TEXT NOT NULL,
    "agent_name" TEXT NOT NULL,
    "litellm_params" JSONB,
    "agent_card_params" JSONB NOT NULL,
    "agent_access_groups" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT NOT NULL,

    CONSTRAINT "LiteLLM_AgentsTable_pkey" PRIMARY KEY ("agent_id")
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
    "object_permission_id" TEXT,
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
    "object_permission_id" TEXT,
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
    "router_settings" JSONB DEFAULT '{}',
    "team_member_permissions" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "policies" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "model_id" INTEGER,

    CONSTRAINT "LiteLLM_TeamTable_pkey" PRIMARY KEY ("team_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_DeletedTeamTable" (
    "id" TEXT NOT NULL,
    "team_id" TEXT NOT NULL,
    "team_alias" TEXT,
    "organization_id" TEXT,
    "object_permission_id" TEXT,
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
    "model_spend" JSONB NOT NULL DEFAULT '{}',
    "model_max_budget" JSONB NOT NULL DEFAULT '{}',
    "router_settings" JSONB DEFAULT '{}',
    "team_member_permissions" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "policies" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "model_id" INTEGER,
    "created_at" TIMESTAMP(3),
    "updated_at" TIMESTAMP(3),
    "deleted_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "deleted_by" TEXT,
    "deleted_by_api_key" TEXT,
    "litellm_changed_by" TEXT,

    CONSTRAINT "LiteLLM_DeletedTeamTable_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_UserTable" (
    "user_id" TEXT NOT NULL,
    "user_alias" TEXT,
    "team_id" TEXT,
    "sso_user_id" TEXT,
    "organization_id" TEXT,
    "object_permission_id" TEXT,
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
    "policies" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "model_spend" JSONB NOT NULL DEFAULT '{}',
    "model_max_budget" JSONB NOT NULL DEFAULT '{}',
    "created_at" TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_UserTable_pkey" PRIMARY KEY ("user_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_ObjectPermissionTable" (
    "object_permission_id" TEXT NOT NULL,
    "mcp_servers" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "mcp_access_groups" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "mcp_tool_permissions" JSONB,
    "vector_stores" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "agents" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "agent_access_groups" TEXT[] DEFAULT ARRAY[]::TEXT[],

    CONSTRAINT "LiteLLM_ObjectPermissionTable_pkey" PRIMARY KEY ("object_permission_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_MCPServerTable" (
    "server_id" TEXT NOT NULL,
    "server_name" TEXT,
    "alias" TEXT,
    "description" TEXT,
    "url" TEXT,
    "transport" TEXT NOT NULL DEFAULT 'sse',
    "auth_type" TEXT,
    "credentials" JSONB DEFAULT '{}',
    "created_at" TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT,
    "mcp_info" JSONB DEFAULT '{}',
    "mcp_access_groups" TEXT[],
    "allowed_tools" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "extra_headers" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "static_headers" JSONB DEFAULT '{}',
    "status" TEXT DEFAULT 'unknown',
    "last_health_check" TIMESTAMP(3),
    "health_check_error" TEXT,
    "command" TEXT,
    "args" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "env" JSONB DEFAULT '{}',
    "authorization_url" TEXT,
    "token_url" TEXT,
    "registration_url" TEXT,
    "allow_all_keys" BOOLEAN NOT NULL DEFAULT false,

    CONSTRAINT "LiteLLM_MCPServerTable_pkey" PRIMARY KEY ("server_id")
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
    "router_settings" JSONB DEFAULT '{}',
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
    "allowed_routes" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "policies" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "model_spend" JSONB NOT NULL DEFAULT '{}',
    "model_max_budget" JSONB NOT NULL DEFAULT '{}',
    "budget_id" TEXT,
    "organization_id" TEXT,
    "object_permission_id" TEXT,
    "created_at" TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT,
    "rotation_count" INTEGER DEFAULT 0,
    "auto_rotate" BOOLEAN DEFAULT false,
    "rotation_interval" TEXT,
    "last_rotation_at" TIMESTAMP(3),
    "key_rotation_at" TIMESTAMP(3),

    CONSTRAINT "LiteLLM_VerificationToken_pkey" PRIMARY KEY ("token")
);

-- CreateTable
CREATE TABLE "LiteLLM_DeletedVerificationToken" (
    "id" TEXT NOT NULL,
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
    "allowed_routes" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "policies" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "model_spend" JSONB NOT NULL DEFAULT '{}',
    "model_max_budget" JSONB NOT NULL DEFAULT '{}',
    "router_settings" JSONB DEFAULT '{}',
    "budget_id" TEXT,
    "organization_id" TEXT,
    "object_permission_id" TEXT,
    "created_at" TIMESTAMP(3),
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3),
    "updated_by" TEXT,
    "rotation_count" INTEGER DEFAULT 0,
    "auto_rotate" BOOLEAN DEFAULT false,
    "rotation_interval" TEXT,
    "last_rotation_at" TIMESTAMP(3),
    "key_rotation_at" TIMESTAMP(3),
    "deleted_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "deleted_by" TEXT,
    "deleted_by_api_key" TEXT,
    "litellm_changed_by" TEXT,

    CONSTRAINT "LiteLLM_DeletedVerificationToken_pkey" PRIMARY KEY ("id")
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
CREATE TABLE "LiteLLM_TagTable" (
    "tag_name" TEXT NOT NULL,
    "description" TEXT,
    "models" TEXT[],
    "model_info" JSONB,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "budget_id" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_TagTable_pkey" PRIMARY KEY ("tag_name")
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
    "organization_id" TEXT,
    "end_user" TEXT,
    "requester_ip_address" TEXT,
    "messages" JSONB DEFAULT '{}',
    "response" JSONB DEFAULT '{}',
    "session_id" TEXT,
    "status" TEXT,
    "mcp_namespaced_tool_name" TEXT,
    "agent_id" TEXT,
    "proxy_server_request" JSONB DEFAULT '{}',

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

-- CreateTable
CREATE TABLE "LiteLLM_DailyUserSpend" (
    "id" TEXT NOT NULL,
    "user_id" TEXT,
    "date" TEXT NOT NULL,
    "api_key" TEXT NOT NULL,
    "model" TEXT,
    "model_group" TEXT,
    "custom_llm_provider" TEXT,
    "mcp_namespaced_tool_name" TEXT,
    "endpoint" TEXT,
    "prompt_tokens" BIGINT NOT NULL DEFAULT 0,
    "completion_tokens" BIGINT NOT NULL DEFAULT 0,
    "cache_read_input_tokens" BIGINT NOT NULL DEFAULT 0,
    "cache_creation_input_tokens" BIGINT NOT NULL DEFAULT 0,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "api_requests" BIGINT NOT NULL DEFAULT 0,
    "successful_requests" BIGINT NOT NULL DEFAULT 0,
    "failed_requests" BIGINT NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_DailyUserSpend_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_DailyOrganizationSpend" (
    "id" TEXT NOT NULL,
    "organization_id" TEXT,
    "date" TEXT NOT NULL,
    "api_key" TEXT NOT NULL,
    "model" TEXT,
    "model_group" TEXT,
    "custom_llm_provider" TEXT,
    "mcp_namespaced_tool_name" TEXT,
    "endpoint" TEXT,
    "prompt_tokens" BIGINT NOT NULL DEFAULT 0,
    "completion_tokens" BIGINT NOT NULL DEFAULT 0,
    "cache_read_input_tokens" BIGINT NOT NULL DEFAULT 0,
    "cache_creation_input_tokens" BIGINT NOT NULL DEFAULT 0,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "api_requests" BIGINT NOT NULL DEFAULT 0,
    "successful_requests" BIGINT NOT NULL DEFAULT 0,
    "failed_requests" BIGINT NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_DailyOrganizationSpend_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_DailyEndUserSpend" (
    "id" TEXT NOT NULL,
    "end_user_id" TEXT,
    "date" TEXT NOT NULL,
    "api_key" TEXT NOT NULL,
    "model" TEXT,
    "model_group" TEXT,
    "custom_llm_provider" TEXT,
    "mcp_namespaced_tool_name" TEXT,
    "endpoint" TEXT,
    "prompt_tokens" BIGINT NOT NULL DEFAULT 0,
    "completion_tokens" BIGINT NOT NULL DEFAULT 0,
    "cache_read_input_tokens" BIGINT NOT NULL DEFAULT 0,
    "cache_creation_input_tokens" BIGINT NOT NULL DEFAULT 0,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "api_requests" BIGINT NOT NULL DEFAULT 0,
    "successful_requests" BIGINT NOT NULL DEFAULT 0,
    "failed_requests" BIGINT NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_DailyEndUserSpend_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_DailyAgentSpend" (
    "id" TEXT NOT NULL,
    "agent_id" TEXT,
    "date" TEXT NOT NULL,
    "api_key" TEXT NOT NULL,
    "model" TEXT,
    "model_group" TEXT,
    "custom_llm_provider" TEXT,
    "mcp_namespaced_tool_name" TEXT,
    "endpoint" TEXT,
    "prompt_tokens" BIGINT NOT NULL DEFAULT 0,
    "completion_tokens" BIGINT NOT NULL DEFAULT 0,
    "cache_read_input_tokens" BIGINT NOT NULL DEFAULT 0,
    "cache_creation_input_tokens" BIGINT NOT NULL DEFAULT 0,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "api_requests" BIGINT NOT NULL DEFAULT 0,
    "successful_requests" BIGINT NOT NULL DEFAULT 0,
    "failed_requests" BIGINT NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_DailyAgentSpend_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_DailyTeamSpend" (
    "id" TEXT NOT NULL,
    "team_id" TEXT,
    "date" TEXT NOT NULL,
    "api_key" TEXT NOT NULL,
    "model" TEXT,
    "model_group" TEXT,
    "custom_llm_provider" TEXT,
    "mcp_namespaced_tool_name" TEXT,
    "endpoint" TEXT,
    "prompt_tokens" BIGINT NOT NULL DEFAULT 0,
    "completion_tokens" BIGINT NOT NULL DEFAULT 0,
    "cache_read_input_tokens" BIGINT NOT NULL DEFAULT 0,
    "cache_creation_input_tokens" BIGINT NOT NULL DEFAULT 0,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "api_requests" BIGINT NOT NULL DEFAULT 0,
    "successful_requests" BIGINT NOT NULL DEFAULT 0,
    "failed_requests" BIGINT NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_DailyTeamSpend_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_DailyTagSpend" (
    "id" TEXT NOT NULL,
    "request_id" TEXT,
    "tag" TEXT,
    "date" TEXT NOT NULL,
    "api_key" TEXT NOT NULL,
    "model" TEXT,
    "model_group" TEXT,
    "custom_llm_provider" TEXT,
    "mcp_namespaced_tool_name" TEXT,
    "endpoint" TEXT,
    "prompt_tokens" BIGINT NOT NULL DEFAULT 0,
    "completion_tokens" BIGINT NOT NULL DEFAULT 0,
    "cache_read_input_tokens" BIGINT NOT NULL DEFAULT 0,
    "cache_creation_input_tokens" BIGINT NOT NULL DEFAULT 0,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "api_requests" BIGINT NOT NULL DEFAULT 0,
    "successful_requests" BIGINT NOT NULL DEFAULT 0,
    "failed_requests" BIGINT NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_DailyTagSpend_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_CronJob" (
    "cronjob_id" TEXT NOT NULL,
    "pod_id" TEXT NOT NULL,
    "status" "JobStatus" NOT NULL DEFAULT 'INACTIVE',
    "last_updated" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "ttl" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_CronJob_pkey" PRIMARY KEY ("cronjob_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_ManagedFileTable" (
    "id" TEXT NOT NULL,
    "unified_file_id" TEXT NOT NULL,
    "file_object" JSONB,
    "model_mappings" JSONB NOT NULL,
    "flat_model_file_ids" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "storage_backend" TEXT,
    "storage_url" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL,
    "updated_by" TEXT,

    CONSTRAINT "LiteLLM_ManagedFileTable_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_ManagedObjectTable" (
    "id" TEXT NOT NULL,
    "unified_object_id" TEXT NOT NULL,
    "model_object_id" TEXT NOT NULL,
    "file_object" JSONB NOT NULL,
    "file_purpose" TEXT NOT NULL,
    "status" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL,
    "updated_by" TEXT,

    CONSTRAINT "LiteLLM_ManagedObjectTable_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_ManagedVectorStoresTable" (
    "vector_store_id" TEXT NOT NULL,
    "custom_llm_provider" TEXT NOT NULL,
    "vector_store_name" TEXT,
    "vector_store_description" TEXT,
    "vector_store_metadata" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,
    "litellm_credential_name" TEXT,
    "litellm_params" JSONB,
    "team_id" TEXT,
    "user_id" TEXT,

    CONSTRAINT "LiteLLM_ManagedVectorStoresTable_pkey" PRIMARY KEY ("vector_store_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_GuardrailsTable" (
    "guardrail_id" TEXT NOT NULL,
    "guardrail_name" TEXT NOT NULL,
    "litellm_params" JSONB NOT NULL,
    "guardrail_info" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_GuardrailsTable_pkey" PRIMARY KEY ("guardrail_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_PromptTable" (
    "id" TEXT NOT NULL,
    "prompt_id" TEXT NOT NULL,
    "version" INTEGER NOT NULL DEFAULT 1,
    "litellm_params" JSONB NOT NULL,
    "prompt_info" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_PromptTable_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_HealthCheckTable" (
    "health_check_id" TEXT NOT NULL,
    "model_name" TEXT NOT NULL,
    "model_id" TEXT,
    "status" TEXT NOT NULL,
    "healthy_count" INTEGER NOT NULL DEFAULT 0,
    "unhealthy_count" INTEGER NOT NULL DEFAULT 0,
    "error_message" TEXT,
    "response_time_ms" DOUBLE PRECISION,
    "details" JSONB,
    "checked_by" TEXT,
    "checked_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_HealthCheckTable_pkey" PRIMARY KEY ("health_check_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_SearchToolsTable" (
    "search_tool_id" TEXT NOT NULL,
    "search_tool_name" TEXT NOT NULL,
    "litellm_params" JSONB NOT NULL,
    "search_tool_info" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_SearchToolsTable_pkey" PRIMARY KEY ("search_tool_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_SSOConfig" (
    "id" TEXT NOT NULL DEFAULT 'sso_config',
    "sso_settings" JSONB NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_SSOConfig_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_ManagedVectorStoreIndexTable" (
    "id" TEXT NOT NULL,
    "index_name" TEXT NOT NULL,
    "litellm_params" JSONB NOT NULL,
    "index_info" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL,
    "updated_by" TEXT,

    CONSTRAINT "LiteLLM_ManagedVectorStoreIndexTable_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_CacheConfig" (
    "id" TEXT NOT NULL DEFAULT 'cache_config',
    "cache_settings" JSONB NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_CacheConfig_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_UISettings" (
    "id" TEXT NOT NULL DEFAULT 'ui_settings',
    "ui_settings" JSONB NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_UISettings_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_SkillsTable" (
    "skill_id" TEXT NOT NULL,
    "display_title" TEXT,
    "description" TEXT,
    "instructions" TEXT,
    "source" TEXT NOT NULL DEFAULT 'custom',
    "latest_version" TEXT,
    "file_content" BYTEA,
    "file_name" TEXT,
    "file_type" TEXT,
    "metadata" JSONB DEFAULT '{}',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT,

    CONSTRAINT "LiteLLM_SkillsTable_pkey" PRIMARY KEY ("skill_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_PolicyTable" (
    "policy_id" TEXT NOT NULL,
    "policy_name" TEXT NOT NULL,
    "inherit" TEXT,
    "description" TEXT,
    "guardrails_add" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "guardrails_remove" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "condition" JSONB DEFAULT '{}',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT,

    CONSTRAINT "LiteLLM_PolicyTable_pkey" PRIMARY KEY ("policy_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_PolicyAttachmentTable" (
    "attachment_id" TEXT NOT NULL,
    "policy_name" TEXT NOT NULL,
    "scope" TEXT,
    "teams" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "keys" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "models" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT,

    CONSTRAINT "LiteLLM_PolicyAttachmentTable_pkey" PRIMARY KEY ("attachment_id")
);

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_CredentialsTable_credential_name_key" ON "LiteLLM_CredentialsTable"("credential_name");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_AgentsTable_agent_name_key" ON "LiteLLM_AgentsTable"("agent_name");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_TeamTable_model_id_key" ON "LiteLLM_TeamTable"("model_id");

-- CreateIndex
CREATE INDEX "LiteLLM_DeletedTeamTable_team_id_idx" ON "LiteLLM_DeletedTeamTable"("team_id");

-- CreateIndex
CREATE INDEX "LiteLLM_DeletedTeamTable_deleted_at_idx" ON "LiteLLM_DeletedTeamTable"("deleted_at");

-- CreateIndex
CREATE INDEX "LiteLLM_DeletedTeamTable_organization_id_idx" ON "LiteLLM_DeletedTeamTable"("organization_id");

-- CreateIndex
CREATE INDEX "LiteLLM_DeletedTeamTable_team_alias_idx" ON "LiteLLM_DeletedTeamTable"("team_alias");

-- CreateIndex
CREATE INDEX "LiteLLM_DeletedTeamTable_created_at_idx" ON "LiteLLM_DeletedTeamTable"("created_at");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_UserTable_sso_user_id_key" ON "LiteLLM_UserTable"("sso_user_id");

-- CreateIndex
CREATE INDEX "LiteLLM_DeletedVerificationToken_token_idx" ON "LiteLLM_DeletedVerificationToken"("token");

-- CreateIndex
CREATE INDEX "LiteLLM_DeletedVerificationToken_deleted_at_idx" ON "LiteLLM_DeletedVerificationToken"("deleted_at");

-- CreateIndex
CREATE INDEX "LiteLLM_DeletedVerificationToken_user_id_idx" ON "LiteLLM_DeletedVerificationToken"("user_id");

-- CreateIndex
CREATE INDEX "LiteLLM_DeletedVerificationToken_team_id_idx" ON "LiteLLM_DeletedVerificationToken"("team_id");

-- CreateIndex
CREATE INDEX "LiteLLM_DeletedVerificationToken_organization_id_idx" ON "LiteLLM_DeletedVerificationToken"("organization_id");

-- CreateIndex
CREATE INDEX "LiteLLM_DeletedVerificationToken_key_alias_idx" ON "LiteLLM_DeletedVerificationToken"("key_alias");

-- CreateIndex
CREATE INDEX "LiteLLM_DeletedVerificationToken_created_at_idx" ON "LiteLLM_DeletedVerificationToken"("created_at");

-- CreateIndex
CREATE INDEX "LiteLLM_SpendLogs_startTime_idx" ON "LiteLLM_SpendLogs"("startTime");

-- CreateIndex
CREATE INDEX "LiteLLM_SpendLogs_end_user_idx" ON "LiteLLM_SpendLogs"("end_user");

-- CreateIndex
CREATE INDEX "LiteLLM_SpendLogs_session_id_idx" ON "LiteLLM_SpendLogs"("session_id");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_OrganizationMembership_user_id_organization_id_key" ON "LiteLLM_OrganizationMembership"("user_id", "organization_id");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyUserSpend_date_idx" ON "LiteLLM_DailyUserSpend"("date");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyUserSpend_user_id_idx" ON "LiteLLM_DailyUserSpend"("user_id");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyUserSpend_api_key_idx" ON "LiteLLM_DailyUserSpend"("api_key");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyUserSpend_model_idx" ON "LiteLLM_DailyUserSpend"("model");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyUserSpend_mcp_namespaced_tool_name_idx" ON "LiteLLM_DailyUserSpend"("mcp_namespaced_tool_name");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyUserSpend_endpoint_idx" ON "LiteLLM_DailyUserSpend"("endpoint");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_DailyUserSpend_user_id_date_api_key_model_custom_ll_key" ON "LiteLLM_DailyUserSpend"("user_id", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyOrganizationSpend_date_idx" ON "LiteLLM_DailyOrganizationSpend"("date");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyOrganizationSpend_organization_id_idx" ON "LiteLLM_DailyOrganizationSpend"("organization_id");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyOrganizationSpend_api_key_idx" ON "LiteLLM_DailyOrganizationSpend"("api_key");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyOrganizationSpend_model_idx" ON "LiteLLM_DailyOrganizationSpend"("model");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyOrganizationSpend_mcp_namespaced_tool_name_idx" ON "LiteLLM_DailyOrganizationSpend"("mcp_namespaced_tool_name");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyOrganizationSpend_endpoint_idx" ON "LiteLLM_DailyOrganizationSpend"("endpoint");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_DailyOrganizationSpend_organization_id_date_api_key_key" ON "LiteLLM_DailyOrganizationSpend"("organization_id", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyEndUserSpend_date_idx" ON "LiteLLM_DailyEndUserSpend"("date");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyEndUserSpend_end_user_id_idx" ON "LiteLLM_DailyEndUserSpend"("end_user_id");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyEndUserSpend_api_key_idx" ON "LiteLLM_DailyEndUserSpend"("api_key");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyEndUserSpend_model_idx" ON "LiteLLM_DailyEndUserSpend"("model");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyEndUserSpend_mcp_namespaced_tool_name_idx" ON "LiteLLM_DailyEndUserSpend"("mcp_namespaced_tool_name");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyEndUserSpend_endpoint_idx" ON "LiteLLM_DailyEndUserSpend"("endpoint");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_DailyEndUserSpend_end_user_id_date_api_key_model_cu_key" ON "LiteLLM_DailyEndUserSpend"("end_user_id", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyAgentSpend_date_idx" ON "LiteLLM_DailyAgentSpend"("date");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyAgentSpend_agent_id_idx" ON "LiteLLM_DailyAgentSpend"("agent_id");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyAgentSpend_api_key_idx" ON "LiteLLM_DailyAgentSpend"("api_key");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyAgentSpend_model_idx" ON "LiteLLM_DailyAgentSpend"("model");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyAgentSpend_mcp_namespaced_tool_name_idx" ON "LiteLLM_DailyAgentSpend"("mcp_namespaced_tool_name");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyAgentSpend_endpoint_idx" ON "LiteLLM_DailyAgentSpend"("endpoint");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_DailyAgentSpend_agent_id_date_api_key_model_custom__key" ON "LiteLLM_DailyAgentSpend"("agent_id", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyTeamSpend_date_idx" ON "LiteLLM_DailyTeamSpend"("date");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyTeamSpend_team_id_idx" ON "LiteLLM_DailyTeamSpend"("team_id");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyTeamSpend_api_key_idx" ON "LiteLLM_DailyTeamSpend"("api_key");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyTeamSpend_model_idx" ON "LiteLLM_DailyTeamSpend"("model");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyTeamSpend_mcp_namespaced_tool_name_idx" ON "LiteLLM_DailyTeamSpend"("mcp_namespaced_tool_name");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyTeamSpend_endpoint_idx" ON "LiteLLM_DailyTeamSpend"("endpoint");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_DailyTeamSpend_team_id_date_api_key_model_custom_ll_key" ON "LiteLLM_DailyTeamSpend"("team_id", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyTagSpend_date_idx" ON "LiteLLM_DailyTagSpend"("date");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyTagSpend_tag_idx" ON "LiteLLM_DailyTagSpend"("tag");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyTagSpend_api_key_idx" ON "LiteLLM_DailyTagSpend"("api_key");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyTagSpend_model_idx" ON "LiteLLM_DailyTagSpend"("model");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyTagSpend_mcp_namespaced_tool_name_idx" ON "LiteLLM_DailyTagSpend"("mcp_namespaced_tool_name");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyTagSpend_endpoint_idx" ON "LiteLLM_DailyTagSpend"("endpoint");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_DailyTagSpend_tag_date_api_key_model_custom_llm_pro_key" ON "LiteLLM_DailyTagSpend"("tag", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_ManagedFileTable_unified_file_id_key" ON "LiteLLM_ManagedFileTable"("unified_file_id");

-- CreateIndex
CREATE INDEX "LiteLLM_ManagedFileTable_unified_file_id_idx" ON "LiteLLM_ManagedFileTable"("unified_file_id");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_ManagedObjectTable_unified_object_id_key" ON "LiteLLM_ManagedObjectTable"("unified_object_id");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_ManagedObjectTable_model_object_id_key" ON "LiteLLM_ManagedObjectTable"("model_object_id");

-- CreateIndex
CREATE INDEX "LiteLLM_ManagedObjectTable_unified_object_id_idx" ON "LiteLLM_ManagedObjectTable"("unified_object_id");

-- CreateIndex
CREATE INDEX "LiteLLM_ManagedObjectTable_model_object_id_idx" ON "LiteLLM_ManagedObjectTable"("model_object_id");

-- CreateIndex
CREATE INDEX "LiteLLM_ManagedVectorStoresTable_team_id_idx" ON "LiteLLM_ManagedVectorStoresTable"("team_id");

-- CreateIndex
CREATE INDEX "LiteLLM_ManagedVectorStoresTable_user_id_idx" ON "LiteLLM_ManagedVectorStoresTable"("user_id");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_GuardrailsTable_guardrail_name_key" ON "LiteLLM_GuardrailsTable"("guardrail_name");

-- CreateIndex
CREATE INDEX "LiteLLM_PromptTable_prompt_id_idx" ON "LiteLLM_PromptTable"("prompt_id");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_PromptTable_prompt_id_version_key" ON "LiteLLM_PromptTable"("prompt_id", "version");

-- CreateIndex
CREATE INDEX "LiteLLM_HealthCheckTable_model_name_idx" ON "LiteLLM_HealthCheckTable"("model_name");

-- CreateIndex
CREATE INDEX "LiteLLM_HealthCheckTable_checked_at_idx" ON "LiteLLM_HealthCheckTable"("checked_at");

-- CreateIndex
CREATE INDEX "LiteLLM_HealthCheckTable_status_idx" ON "LiteLLM_HealthCheckTable"("status");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_SearchToolsTable_search_tool_name_key" ON "LiteLLM_SearchToolsTable"("search_tool_name");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_ManagedVectorStoreIndexTable_index_name_key" ON "LiteLLM_ManagedVectorStoreIndexTable"("index_name");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_PolicyTable_policy_name_key" ON "LiteLLM_PolicyTable"("policy_name");

-- AddForeignKey
ALTER TABLE "LiteLLM_OrganizationTable" ADD CONSTRAINT "LiteLLM_OrganizationTable_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_OrganizationTable" ADD CONSTRAINT "LiteLLM_OrganizationTable_object_permission_id_fkey" FOREIGN KEY ("object_permission_id") REFERENCES "LiteLLM_ObjectPermissionTable"("object_permission_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_TeamTable" ADD CONSTRAINT "LiteLLM_TeamTable_organization_id_fkey" FOREIGN KEY ("organization_id") REFERENCES "LiteLLM_OrganizationTable"("organization_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_TeamTable" ADD CONSTRAINT "LiteLLM_TeamTable_model_id_fkey" FOREIGN KEY ("model_id") REFERENCES "LiteLLM_ModelTable"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_TeamTable" ADD CONSTRAINT "LiteLLM_TeamTable_object_permission_id_fkey" FOREIGN KEY ("object_permission_id") REFERENCES "LiteLLM_ObjectPermissionTable"("object_permission_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_UserTable" ADD CONSTRAINT "LiteLLM_UserTable_organization_id_fkey" FOREIGN KEY ("organization_id") REFERENCES "LiteLLM_OrganizationTable"("organization_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_UserTable" ADD CONSTRAINT "LiteLLM_UserTable_object_permission_id_fkey" FOREIGN KEY ("object_permission_id") REFERENCES "LiteLLM_ObjectPermissionTable"("object_permission_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_VerificationToken" ADD CONSTRAINT "LiteLLM_VerificationToken_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_VerificationToken" ADD CONSTRAINT "LiteLLM_VerificationToken_organization_id_fkey" FOREIGN KEY ("organization_id") REFERENCES "LiteLLM_OrganizationTable"("organization_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_VerificationToken" ADD CONSTRAINT "LiteLLM_VerificationToken_object_permission_id_fkey" FOREIGN KEY ("object_permission_id") REFERENCES "LiteLLM_ObjectPermissionTable"("object_permission_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_EndUserTable" ADD CONSTRAINT "LiteLLM_EndUserTable_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_TagTable" ADD CONSTRAINT "LiteLLM_TagTable_budget_id_fkey" FOREIGN KEY ("budget_id") REFERENCES "LiteLLM_BudgetTable"("budget_id") ON DELETE SET NULL ON UPDATE CASCADE;

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

