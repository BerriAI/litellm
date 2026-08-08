CREATE TABLE IF NOT EXISTS "LiteLLM_ShadowEvalJob" (
    "id" TEXT NOT NULL,
    "team_id" TEXT,
    "organization_id" TEXT,
    "api_key_id" TEXT NOT NULL,
    "router_name" TEXT NOT NULL,
    "shadow_percentage" DOUBLE PRECISION NOT NULL,
    "judge_model" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "request_count" INTEGER NOT NULL DEFAULT 0,
    "completed_count" INTEGER NOT NULL DEFAULT 0,
    "failed_count" INTEGER NOT NULL DEFAULT 0,
    "result_json" JSONB,
    "cost_estimate" DOUBLE PRECISION,
    "cost_actual" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "completed_at" TIMESTAMP(3),

    CONSTRAINT "LiteLLM_ShadowEvalJob_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "LiteLLM_ShadowEvalJob_team_id_status_idx" ON "LiteLLM_ShadowEvalJob"("team_id", "status");
CREATE INDEX IF NOT EXISTS "LiteLLM_ShadowEvalJob_api_key_id_status_idx" ON "LiteLLM_ShadowEvalJob"("api_key_id", "status");
CREATE INDEX IF NOT EXISTS "LiteLLM_ShadowEvalJob_created_at_idx" ON "LiteLLM_ShadowEvalJob"("created_at");

CREATE TABLE IF NOT EXISTS "LiteLLM_ShadowEvalVerdict" (
    "id" TEXT NOT NULL,
    "job_id" TEXT NOT NULL,
    "request_id" TEXT NOT NULL,
    "shadow_request_id" TEXT,
    "tier_classification" TEXT,
    "real_model" TEXT NOT NULL,
    "shadow_model" TEXT NOT NULL,
    "real_response_tokens" INTEGER,
    "shadow_response_tokens" INTEGER,
    "judge_preference" TEXT NOT NULL,
    "judge_confidence" DOUBLE PRECISION,
    "judge_reasoning" TEXT,
    "judge_model" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_ShadowEvalVerdict_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "LiteLLM_ShadowEvalVerdict_job_id_idx" ON "LiteLLM_ShadowEvalVerdict"("job_id");
CREATE INDEX IF NOT EXISTS "LiteLLM_ShadowEvalVerdict_request_id_idx" ON "LiteLLM_ShadowEvalVerdict"("request_id");
