SET LOCAL lock_timeout = '10s';

CREATE TABLE IF NOT EXISTS "LiteLLM_PerformanceAlertState" (
    "id" INTEGER PRIMARY KEY,
    "owner" VARCHAR(32) NOT NULL,
    "next_run_at" TIMESTAMPTZ(3) NOT NULL,
    "day" VARCHAR(10),
    "scope" VARCHAR(16),
    "resume_after" VARCHAR(1024) NOT NULL DEFAULT '',
    "wrapped" BOOLEAN NOT NULL DEFAULT false,
    "checked" INTEGER NOT NULL DEFAULT 0,
    "skipped" INTEGER NOT NULL DEFAULT 0,
    "start_scope" VARCHAR(16),
    "start_after" VARCHAR(1024) NOT NULL DEFAULT '',
    "completed_at" TIMESTAMP(3),
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE "LiteLLM_DailyModelUsage"
    ADD COLUMN IF NOT EXISTS "total_response_time_ms" BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS "timed_requests" BIGINT NOT NULL DEFAULT 0;
