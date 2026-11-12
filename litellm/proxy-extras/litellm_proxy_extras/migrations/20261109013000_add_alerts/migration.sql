SET LOCAL lock_timeout = '10s';

CREATE TABLE IF NOT EXISTS "LiteLLM_Alert" (
    "id" BIGSERIAL PRIMARY KEY,
    "detection_key" VARCHAR(64) NOT NULL,
    "type" VARCHAR(32) NOT NULL,
    "resource_type" VARCHAR(16) NOT NULL,
    "resource_id" VARCHAR(1024) NOT NULL,
    "status" VARCHAR(16) NOT NULL DEFAULT 'open',
    "window_start" TIMESTAMP(3) NOT NULL,
    "window_end" TIMESTAMP(3) NOT NULL,
    "evidence" JSONB NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" VARCHAR(1024),
    CHECK ("status" IN ('open', 'resolved', 'dismissed')),
    CHECK ("window_end" > "window_start"),
    CHECK (octet_length("evidence"::text) <= 8192)
);

CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_Alert_detection_key_key" ON "LiteLLM_Alert"("detection_key");
CREATE INDEX IF NOT EXISTS "LiteLLM_Alert_status_id_idx" ON "LiteLLM_Alert"("status", "id");
