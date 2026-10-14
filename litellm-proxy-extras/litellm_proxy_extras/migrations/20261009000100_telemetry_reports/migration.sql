CREATE TABLE IF NOT EXISTS "LiteLLM_TelemetryReport" (
    "id" TEXT NOT NULL,
    "window_start" DOUBLE PRECISION NOT NULL,
    "window_end" DOUBLE PRECISION NOT NULL,
    "report" JSONB NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "LiteLLM_TelemetryReport_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "LiteLLM_TelemetryReport_window_end_idx" ON "LiteLLM_TelemetryReport"("window_end");

CREATE INDEX IF NOT EXISTS "LiteLLM_TelemetryReport_created_at_idx" ON "LiteLLM_TelemetryReport"("created_at");
