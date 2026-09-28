CREATE TABLE "LiteLLM_DailyModelUsage" (
    "date" TEXT NOT NULL,
    "model_group" TEXT NOT NULL,
    "model" TEXT NOT NULL,
    "custom_llm_provider" TEXT NOT NULL,
    "task_type" TEXT NOT NULL,
    "spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "prompt_tokens" BIGINT NOT NULL DEFAULT 0,
    "completion_tokens" BIGINT NOT NULL DEFAULT 0,
    "request_count" BIGINT NOT NULL DEFAULT 0,
    "successful_requests" BIGINT NOT NULL DEFAULT 0,
    "failed_requests" BIGINT NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,
    CONSTRAINT "LiteLLM_DailyModelUsage_pkey" PRIMARY KEY ("date", "model_group", "model", "custom_llm_provider", "task_type")
);

CREATE INDEX "LiteLLM_DailyModelUsage_date_idx" ON "LiteLLM_DailyModelUsage"("date");
CREATE INDEX "LiteLLM_DailyModelUsage_model_group_idx" ON "LiteLLM_DailyModelUsage"("model_group");
