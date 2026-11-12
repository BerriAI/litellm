SET LOCAL lock_timeout = '10s';

CREATE TABLE IF NOT EXISTS "LiteLLM_TaskObservation" (
    "delivery_id" TEXT NOT NULL,
    "request_id" TEXT NOT NULL,
    "is_batch" BOOLEAN NOT NULL,
    "start_time" TIMESTAMP(3) NOT NULL,
    "team_id" TEXT,
    "model_group" TEXT,
    "model" TEXT,
    "model_config_id" TEXT,
    "session_id" TEXT,
    "spend" DOUBLE PRECISION,
    "latency_ms" DOUBLE PRECISION,
    "status" TEXT,
    "request_text" TEXT,
    "context_text" TEXT,
    "text_hash" TEXT,
    "excluded" BOOLEAN NOT NULL DEFAULT false,
    "inference" BOOLEAN NOT NULL DEFAULT true,
    "observed_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "LiteLLM_TaskObservation_pkey" PRIMARY KEY ("delivery_id")
);
CREATE INDEX IF NOT EXISTS "LiteLLM_TaskObservation_request_id_idx" ON "LiteLLM_TaskObservation"("request_id");
CREATE INDEX IF NOT EXISTS "LiteLLM_TaskObservation_start_time_idx" ON "LiteLLM_TaskObservation"("start_time");
CREATE INDEX IF NOT EXISTS "LiteLLM_TaskObservation_text_hash_idx" ON "LiteLLM_TaskObservation"("text_hash");

CREATE TABLE IF NOT EXISTS "LiteLLM_TaskDefinition" (
    "task_id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "inclusion_criteria" TEXT NOT NULL,
    "exclusion_criteria" TEXT NOT NULL,
    "examples" JSONB NOT NULL DEFAULT '[]',
    "required_capabilities" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "status" TEXT NOT NULL,
    "source" TEXT NOT NULL,
    "map_x" DOUBLE PRECISION NOT NULL,
    "map_y" DOUBLE PRECISION NOT NULL,
    "discovery_size" INTEGER NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "frozen_at" TIMESTAMP(3),
    CONSTRAINT "LiteLLM_TaskDefinition_pkey" PRIMARY KEY ("task_id")
);
CREATE INDEX IF NOT EXISTS "LiteLLM_TaskDefinition_status_idx" ON "LiteLLM_TaskDefinition"("status");

CREATE TABLE IF NOT EXISTS "LiteLLM_TaskAssignment" (
    "request_id" TEXT NOT NULL,
    "task_id" TEXT,
    "outcome" TEXT NOT NULL,
    "text_hash" TEXT,
    "probabilities" JSONB NOT NULL DEFAULT '{}',
    "top_probability" DOUBLE PRECISION,
    "fit_probability" DOUBLE PRECISION,
    "decision_model" TEXT NOT NULL,
    "decision_provenance" JSONB NOT NULL DEFAULT '{}',
    "attempts" INTEGER NOT NULL DEFAULT 0,
    "error" TEXT,
    "assigned_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "LiteLLM_TaskAssignment_pkey" PRIMARY KEY ("request_id")
);
CREATE INDEX IF NOT EXISTS "LiteLLM_TaskAssignment_task_id_idx" ON "LiteLLM_TaskAssignment"("task_id");
CREATE INDEX IF NOT EXISTS "LiteLLM_TaskAssignment_text_hash_idx" ON "LiteLLM_TaskAssignment"("text_hash");

CREATE TABLE IF NOT EXISTS "LiteLLM_TaskMapPoint" (
    "request_id" TEXT NOT NULL,
    "status" TEXT NOT NULL,
    "task_id" TEXT,
    "x" DOUBLE PRECISION NOT NULL,
    "y" DOUBLE PRECISION NOT NULL,
    CONSTRAINT "LiteLLM_TaskMapPoint_pkey" PRIMARY KEY ("request_id","status")
);

CREATE TABLE IF NOT EXISTS "LiteLLM_TaskInsightsState" (
    "id" TEXT NOT NULL DEFAULT 'default',
    "discovery_status" TEXT NOT NULL DEFAULT 'idle',
    "discovery_requested_at" TIMESTAMP(3),
    "discovery_window_start" TIMESTAMP(3),
    "discovery_window_end" TIMESTAMP(3),
    "discovery_max_requests" INTEGER NOT NULL DEFAULT 500,
    "discovery_finished_at" TIMESTAMP(3),
    "discovery_error" TEXT,
    "discovery_provenance" JSONB NOT NULL DEFAULT '{}',
    "taxonomy_frozen_at" TIMESTAMP(3),
    "last_bucketed_at" TIMESTAMP(3),
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "LiteLLM_TaskInsightsState_pkey" PRIMARY KEY ("id")
);
