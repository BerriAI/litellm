-- CreateTable
CREATE TABLE "LiteLLM_HealthCheckTable" (
    "health_check_id" TEXT NOT NULL,
    "model_name" TEXT NOT NULL,
    "model_id" TEXT,
    "status" TEXT NOT NULL,
    "healthy_count" INTEGER NOT NULL DEFAULT 0,
    "unhealthy_count" INTEGER NOT NULL DEFAULT 0,
    "error_message" TEXT,
    "response_time_ms" DOUBLE PRECISION,
    "details" JSONB,
    "checked_by" TEXT,
    "checked_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_HealthCheckTable_pkey" PRIMARY KEY ("health_check_id")
);

-- CreateIndex
CREATE INDEX "LiteLLM_HealthCheckTable_model_name_idx" ON "LiteLLM_HealthCheckTable"("model_name");

-- CreateIndex
CREATE INDEX "LiteLLM_HealthCheckTable_checked_at_idx" ON "LiteLLM_HealthCheckTable"("checked_at");

-- CreateIndex
CREATE INDEX "LiteLLM_HealthCheckTable_status_idx" ON "LiteLLM_HealthCheckTable"("status");

