CREATE TABLE "ExternalAgentSource" (
    "source_id" TEXT NOT NULL,
    "vendor" TEXT NOT NULL DEFAULT 'grok_bot',
    "display_name" TEXT NOT NULL,
    "tenant_id" TEXT,
    "signing_secret" TEXT NOT NULL,
    "previous_signing_secret" TEXT,
    "mode" TEXT NOT NULL DEFAULT 'shadow',
    "default_policy_id" TEXT,
    "created_by" TEXT NOT NULL,
    "updated_by" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ExternalAgentSource_pkey" PRIMARY KEY ("source_id")
);

CREATE TABLE "ExternalAgent" (
    "id" TEXT NOT NULL,
    "source_id" TEXT NOT NULL,
    "external_agent_id" TEXT NOT NULL,
    "display_name" TEXT,
    "team_id" TEXT,
    "owner_user_id" TEXT,
    "owner_inferred" BOOLEAN NOT NULL DEFAULT false,
    "status" TEXT NOT NULL DEFAULT 'active',
    "first_seen_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "last_seen_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "request_count" INTEGER NOT NULL DEFAULT 0,
    "last_actor_id" TEXT,
    "last_model" TEXT,

    CONSTRAINT "ExternalAgent_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "ExternalAgentActor" (
    "id" TEXT NOT NULL,
    "source_id" TEXT NOT NULL,
    "external_actor_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,

    CONSTRAINT "ExternalAgentActor_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "ExternalAgentPolicy" (
    "policy_id" TEXT NOT NULL,
    "source_id" TEXT NOT NULL,
    "team_id" TEXT,
    "blocked_tools" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "allow_unmapped_actors" BOOLEAN NOT NULL DEFAULT true,
    "require_owner" BOOLEAN NOT NULL DEFAULT false,

    CONSTRAINT "ExternalAgentPolicy_pkey" PRIMARY KEY ("policy_id")
);

CREATE TABLE "ExternalAgentDecision" (
    "request_id" TEXT NOT NULL,
    "source_id" TEXT NOT NULL,
    "external_agent_id" TEXT NOT NULL,
    "actor_id" TEXT,
    "user_id" TEXT,
    "team_id" TEXT,
    "frame_type" TEXT NOT NULL,
    "tool_names" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "verdict" TEXT NOT NULL,
    "would_verdict" TEXT NOT NULL,
    "reason" TEXT,
    "enforced" BOOLEAN NOT NULL,
    "policy_id" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ExternalAgentDecision_pkey" PRIMARY KEY ("request_id")
);

CREATE INDEX "ExternalAgent_source_id_team_id_idx" ON "ExternalAgent"("source_id", "team_id");

CREATE UNIQUE INDEX "ExternalAgent_source_id_external_agent_id_key" ON "ExternalAgent"("source_id", "external_agent_id");

CREATE UNIQUE INDEX "ExternalAgentActor_source_id_external_actor_id_key" ON "ExternalAgentActor"("source_id", "external_actor_id");

CREATE UNIQUE INDEX "ExternalAgentPolicy_source_id_team_id_key" ON "ExternalAgentPolicy"("source_id", "team_id");

CREATE INDEX "ExternalAgentDecision_source_id_created_at_idx" ON "ExternalAgentDecision"("source_id", "created_at");

CREATE INDEX "ExternalAgentDecision_source_id_external_agent_id_created_a_idx" ON "ExternalAgentDecision"("source_id", "external_agent_id", "created_at");
