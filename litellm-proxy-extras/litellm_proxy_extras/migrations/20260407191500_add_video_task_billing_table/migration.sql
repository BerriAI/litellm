CREATE TABLE IF NOT EXISTS "LiteLLM_VideoTaskTable" (
    "video_id" TEXT NOT NULL,
    "provider_task_id" TEXT NOT NULL,
    "api_key" TEXT NOT NULL DEFAULT '',
    "user" TEXT DEFAULT '',
    "team_id" TEXT,
    "organization_id" TEXT,
    "end_user" TEXT,
    "custom_llm_provider" TEXT DEFAULT '',
    "model" TEXT NOT NULL DEFAULT '',
    "model_group" TEXT DEFAULT '',
    "model_id" TEXT DEFAULT '',
    "provider_model" TEXT DEFAULT '',
    "pricing_model" TEXT NOT NULL DEFAULT '',
    "pricing_currency" TEXT NOT NULL DEFAULT 'CNY',
    "price_per_million_tokens" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "has_input_video" BOOLEAN NOT NULL DEFAULT false,
    "provider_status" TEXT NOT NULL DEFAULT 'queued',
    "billing_state" TEXT NOT NULL DEFAULT 'pending',
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "total_tokens" BIGINT NOT NULL DEFAULT 0,
    "prompt_tokens" BIGINT NOT NULL DEFAULT 0,
    "completion_tokens" BIGINT NOT NULL DEFAULT 0,
    "duration_seconds" DOUBLE PRECISION,
    "request_tags" JSONB DEFAULT '[]',
    "metadata" JSONB DEFAULT '{}',
    "completed_at" TIMESTAMP(3),
    "billed_at" TIMESTAMP(3),
    "last_checked_at" TIMESTAMP(3),
    "next_check_at" TIMESTAMP(3),
    "check_attempts" INTEGER NOT NULL DEFAULT 0,
    "last_error" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "LiteLLM_VideoTaskTable_pkey" PRIMARY KEY ("video_id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_VideoTaskTable_provider_task_id_key"
ON "LiteLLM_VideoTaskTable"("provider_task_id");

CREATE INDEX IF NOT EXISTS "LiteLLM_VideoTaskTable_billing_state_next_check_at_idx"
ON "LiteLLM_VideoTaskTable"("billing_state", "next_check_at");

CREATE INDEX IF NOT EXISTS "LiteLLM_VideoTaskTable_api_key_idx"
ON "LiteLLM_VideoTaskTable"("api_key");

CREATE INDEX IF NOT EXISTS "LiteLLM_VideoTaskTable_team_id_idx"
ON "LiteLLM_VideoTaskTable"("team_id");

CREATE INDEX IF NOT EXISTS "LiteLLM_VideoTaskTable_organization_id_idx"
ON "LiteLLM_VideoTaskTable"("organization_id");

CREATE INDEX IF NOT EXISTS "LiteLLM_VideoTaskTable_model_idx"
ON "LiteLLM_VideoTaskTable"("model");

CREATE INDEX IF NOT EXISTS "LiteLLM_VideoTaskTable_model_group_idx"
ON "LiteLLM_VideoTaskTable"("model_group");
