CREATE TABLE IF NOT EXISTS "LiteLLM_SessionReplayJob" (
    "id" TEXT NOT NULL,
    "session_id" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'running',
    "arms" JSONB NOT NULL,
    "judge_model" TEXT NOT NULL,
    "max_turns" INTEGER NOT NULL,
    "source_request_id" TEXT,
    "turns_completed" INTEGER NOT NULL DEFAULT 0,
    "result" JSONB,
    "error" TEXT,
    "created_by" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "finished_at" TIMESTAMP(3),

    CONSTRAINT "LiteLLM_SessionReplayJob_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "LiteLLM_SessionReplayJob_session_id_idx" ON "LiteLLM_SessionReplayJob"("session_id");

CREATE INDEX IF NOT EXISTS "LiteLLM_SessionReplayJob_created_at_idx" ON "LiteLLM_SessionReplayJob"("created_at");

-- A replay issues real billable calls for every recorded turn on every arm, so two
-- concurrent starts on one session double that spend for no extra signal. Partial
-- indexes are not expressible in schema.prisma, so this lives here only.
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_SessionReplayJob_one_active_per_session"
    ON "LiteLLM_SessionReplayJob"("session_id") WHERE "status" = 'running';
