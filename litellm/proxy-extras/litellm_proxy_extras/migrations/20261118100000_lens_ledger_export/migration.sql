SET LOCAL lock_timeout = '10s';
CREATE TABLE IF NOT EXISTS "LiteLLM_LensLedgerExport" (
    "event_id" TEXT PRIMARY KEY,
    "request_id" TEXT NOT NULL,
    "event" JSONB NOT NULL,
    "attempts" INTEGER NOT NULL DEFAULT 0,
    "next_attempt_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "lease_owner" TEXT,
    "lease_until" TIMESTAMP(3),
    "last_error" TEXT,
    "parked_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "LiteLLM_LensLedgerExport_parked_at_next_attempt_at_idx"
    ON "LiteLLM_LensLedgerExport" ("parked_at", "next_attempt_at");
CREATE TABLE IF NOT EXISTS "LiteLLM_LensLedgerPayload" (
    "event_id" TEXT NOT NULL,
    "field" TEXT NOT NULL,
    "team_id" TEXT NOT NULL,
    "api_key" TEXT NOT NULL,
    "event_time_ms" BIGINT NOT NULL,
    "content" TEXT NOT NULL,
    "bytes" INTEGER NOT NULL,
    "sha256" TEXT NOT NULL,
    "attempts" INTEGER NOT NULL DEFAULT 0,
    "next_attempt_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "lease_owner" TEXT,
    "lease_until" TIMESTAMP(3),
    "last_error" TEXT,
    "parked_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "LiteLLM_LensLedgerPayload_pkey" PRIMARY KEY ("event_id", "field")
);
CREATE INDEX IF NOT EXISTS "LiteLLM_LensLedgerPayload_parked_at_next_attempt_at_idx"
    ON "LiteLLM_LensLedgerPayload" ("parked_at", "next_attempt_at");
