SET LOCAL lock_timeout = '10s';

CREATE TYPE "AccountingEventKind" AS ENUM ('inbound_usage', 'outbound_usage', 'cost');

CREATE TYPE "AccountingDeliveryState" AS ENUM ('pending', 'applied', 'parked');

CREATE TYPE "AccountingDeliveryTarget" AS ENUM ('LiteLLM_VerificationToken', 'LiteLLM_UserTable', 'LiteLLM_TeamTable', 'LiteLLM_TeamMembership', 'LiteLLM_OrganizationTable', 'LiteLLM_OrganizationMembership', 'LiteLLM_EndUserTable', 'LiteLLM_AgentsTable', 'LiteLLM_TagTable', 'LiteLLM_ProjectTable', 'LiteLLM_ModelAccessGroupBudgetTable', 'LiteLLM_BudgetTable', 'LiteLLM_BudgetWindowSpend', 'LiteLLM_DailyUserSpend', 'LiteLLM_DailyTeamSpend', 'LiteLLM_DailyOrganizationSpend', 'LiteLLM_DailyEndUserSpend', 'LiteLLM_DailyAgentSpend', 'LiteLLM_DailyTagSpend', 'LiteLLM_DailyModelUsage', 'LiteLLM_DailyToolSpend', 'LiteLLM_AutoRouterDailySpend');

-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_AccountingEvent" (
    "event_id" TEXT NOT NULL,
    "kind" "AccountingEventKind" NOT NULL,
    "source_event_id" TEXT,
    "schema_version" INTEGER NOT NULL,
    "payload_hash" TEXT NOT NULL,
    "occurred_at" TIMESTAMP(3) NOT NULL,
    "recorded_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "payload" JSONB NOT NULL,

    CONSTRAINT "LiteLLM_AccountingEvent_schema_version_check" CHECK ("schema_version" > 0),
    CONSTRAINT "LiteLLM_AccountingEvent_pkey" PRIMARY KEY ("event_id")
);

-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_AccountingDelivery" (
    "event_id" TEXT NOT NULL,
    "target" "AccountingDeliveryTarget" NOT NULL,
    "version" INTEGER NOT NULL,
    "state" "AccountingDeliveryState" NOT NULL DEFAULT 'pending',
    "attempts" INTEGER NOT NULL DEFAULT 0,
    "next_attempt_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "applied_at" TIMESTAMP(3),
    "last_error" TEXT,

    CONSTRAINT "LiteLLM_AccountingDelivery_version_check" CHECK ("version" > 0),
    CONSTRAINT "LiteLLM_AccountingDelivery_attempts_check" CHECK ("attempts" >= 0),
    CONSTRAINT "LiteLLM_AccountingDelivery_pkey" PRIMARY KEY ("event_id","target","version")
);

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_AccountingEvent_recorded_at_idx" ON "LiteLLM_AccountingEvent"("recorded_at");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_AccountingEvent_kind_source_event_id_idx" ON "LiteLLM_AccountingEvent"("kind", "source_event_id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_AccountingDelivery_target_state_next_attempt_at_idx" ON "LiteLLM_AccountingDelivery"("target", "state", "next_attempt_at");
