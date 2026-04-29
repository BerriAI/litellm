-- CreateTable
CREATE TABLE "LiteLLM_ScheduledTaskTable" (
    "task_id" TEXT NOT NULL,
    "owner_token" TEXT NOT NULL,
    "user_id" TEXT,
    "team_id" TEXT,
    "agent_id" TEXT,
    "title" TEXT NOT NULL,
    "action" TEXT NOT NULL,
    "action_args" JSONB,
    "check_prompt" TEXT,
    "format_prompt" TEXT,
    "schedule_kind" TEXT NOT NULL,
    "schedule_spec" TEXT NOT NULL,
    "schedule_tz" TEXT,
    "next_run_at" TIMESTAMP(3) NOT NULL,
    "expires_at" TIMESTAMP(3) NOT NULL,
    "fire_once" BOOLEAN NOT NULL DEFAULT true,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "last_fired_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_ScheduledTaskTable_pkey" PRIMARY KEY ("task_id")
);

-- CreateIndex
CREATE INDEX "LiteLLM_ScheduledTaskTable_owner_token_status_idx"
    ON "LiteLLM_ScheduledTaskTable" ("owner_token", "status");

-- CreateIndex
CREATE INDEX "LiteLLM_ScheduledTaskTable_user_id_idx"
    ON "LiteLLM_ScheduledTaskTable" ("user_id");

-- CreateIndex
CREATE INDEX "LiteLLM_ScheduledTaskTable_team_id_idx"
    ON "LiteLLM_ScheduledTaskTable" ("team_id");

-- CreateIndex
CREATE INDEX "LiteLLM_ScheduledTaskTable_agent_id_status_idx"
    ON "LiteLLM_ScheduledTaskTable" ("agent_id", "status");

-- Hand-appended (Prisma cannot express partial indexes or CHECK constraints):

-- Partial index over rows the ticker actually scans.
CREATE INDEX "LiteLLM_ScheduledTaskTable_next_run_due_idx"
    ON "LiteLLM_ScheduledTaskTable" ("next_run_at")
    WHERE status = 'pending';

-- Constrain enums at DB. Bad writes bounce.
ALTER TABLE "LiteLLM_ScheduledTaskTable"
    ADD CONSTRAINT "LiteLLM_ScheduledTaskTable_schedule_kind_check"
    CHECK (schedule_kind IN ('interval','cron','once'));
ALTER TABLE "LiteLLM_ScheduledTaskTable"
    ADD CONSTRAINT "LiteLLM_ScheduledTaskTable_status_check"
    CHECK (status IN ('pending','fired','expired','cancelled'));
ALTER TABLE "LiteLLM_ScheduledTaskTable"
    ADD CONSTRAINT "LiteLLM_ScheduledTaskTable_action_prompt_check"
    CHECK (action <> 'check' OR check_prompt IS NOT NULL);
