SET LOCAL lock_timeout = '10s';

CREATE TYPE "ScheduledJobRunStatus" AS ENUM ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELLED');

CREATE TABLE IF NOT EXISTS "LiteLLM_ScheduledJobRun" (
    "id" TEXT NOT NULL,
    "job_name" TEXT NOT NULL,
    "lock_group" TEXT NOT NULL,
    "schedule" TEXT NOT NULL,
    "scheduled_for" TIMESTAMPTZ(3) NOT NULL,
    "status" "ScheduledJobRunStatus" NOT NULL DEFAULT 'PENDING',
    "claimed_by" TEXT,
    "claimed_at" TIMESTAMPTZ(3),
    "lease_expires_at" TIMESTAMPTZ(3),
    "finished_at" TIMESTAMPTZ(3),
    "created_at" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_ScheduledJobRun_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "LiteLLM_ScheduledJobRun_job_name_status_idx" ON "LiteLLM_ScheduledJobRun"("job_name", "status");

CREATE INDEX IF NOT EXISTS "LiteLLM_ScheduledJobRun_lock_group_status_idx" ON "LiteLLM_ScheduledJobRun"("lock_group", "status");
