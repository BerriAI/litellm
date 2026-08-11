-- CreateTable
CREATE TABLE "LiteLLM_ShadowEvalJob" (
    "id" TEXT NOT NULL,
    "api_key_id" TEXT NOT NULL,
    "router_name" TEXT NOT NULL,
    "shadow_percentage" DOUBLE PRECISION NOT NULL,
    "judge_model" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "request_count" INTEGER NOT NULL DEFAULT 0,
    "completed_count" INTEGER NOT NULL DEFAULT 0,
    "failed_count" INTEGER NOT NULL DEFAULT 0,
    "last_error" TEXT,
    "cost_estimate" DOUBLE PRECISION,
    "cost_actual" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "ends_at" TIMESTAMP(3),
    "completed_at" TIMESTAMP(3),

    CONSTRAINT "LiteLLM_ShadowEvalJob_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_ShadowEvalVerdict" (
    "id" TEXT NOT NULL,
    "job_id" TEXT NOT NULL,
    "request_id" TEXT NOT NULL,
    "tier_classification" TEXT,
    "real_model" TEXT NOT NULL,
    "shadow_model" TEXT NOT NULL,
    "judge_preference" TEXT NOT NULL,
    "judge_confidence" DOUBLE PRECISION,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_ShadowEvalVerdict_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "LiteLLM_ShadowEvalJob_api_key_id_status_idx" ON "LiteLLM_ShadowEvalJob"("api_key_id", "status");

-- CreateIndex
CREATE INDEX "LiteLLM_ShadowEvalJob_status_idx" ON "LiteLLM_ShadowEvalJob"("status");

-- CreateIndex
CREATE INDEX "LiteLLM_ShadowEvalJob_created_at_idx" ON "LiteLLM_ShadowEvalJob"("created_at");

-- CreateIndex
CREATE INDEX "LiteLLM_ShadowEvalVerdict_job_id_idx" ON "LiteLLM_ShadowEvalVerdict"("job_id");

-- One active job per key, enforced by the database rather than a read-then-create in
-- the start endpoint, which races against a concurrent start on another pod. Partial
-- indexes are not expressible in schema.prisma, so this lives here only.
CREATE UNIQUE INDEX "LiteLLM_ShadowEvalJob_one_active_per_key"
    ON "LiteLLM_ShadowEvalJob"("api_key_id") WHERE status IN ('pending', 'running');

