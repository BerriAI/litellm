-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_DailyRequestErrors" (
    "date" TEXT NOT NULL,
    "api_key" TEXT NOT NULL,
    "team_id" TEXT NOT NULL DEFAULT '',
    "user_id" TEXT NOT NULL DEFAULT '',
    "model_group" TEXT NOT NULL DEFAULT '',
    "status_code" INTEGER NOT NULL,
    "failed_requests" BIGINT NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LiteLLM_DailyRequestErrors_pkey" PRIMARY KEY ("date","api_key","team_id","user_id","model_group","status_code")
);

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_DailyRequestErrors_date_idx" ON "LiteLLM_DailyRequestErrors"("date");
