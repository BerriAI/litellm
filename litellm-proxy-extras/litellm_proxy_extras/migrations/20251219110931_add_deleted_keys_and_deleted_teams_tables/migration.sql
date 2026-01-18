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
    "team_member_permissions" TEXT[] DEFAULT ARRAY[]::TEXT[],
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
    "model_spend" JSONB NOT NULL DEFAULT '{}',
    "model_max_budget" JSONB NOT NULL DEFAULT '{}',
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

