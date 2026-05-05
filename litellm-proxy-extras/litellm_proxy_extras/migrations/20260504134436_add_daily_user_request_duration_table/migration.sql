-- CreateTable
CREATE TABLE "LiteLLM_DailyUserRequestDuration" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "date" TEXT NOT NULL,
    "total_request_duration_ms" BIGINT NOT NULL DEFAULT 0,
    "api_requests" BIGINT NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_DailyUserRequestDuration_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_DailyUserRequestDuration_user_id_date_key" ON "LiteLLM_DailyUserRequestDuration"("user_id", "date");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyUserRequestDuration_date_idx" ON "LiteLLM_DailyUserRequestDuration"("date");

-- CreateIndex
CREATE INDEX "LiteLLM_DailyUserRequestDuration_user_id_date_idx" ON "LiteLLM_DailyUserRequestDuration"("user_id", "date");
