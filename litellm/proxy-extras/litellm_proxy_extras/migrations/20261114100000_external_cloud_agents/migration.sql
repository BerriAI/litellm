SET LOCAL lock_timeout = '10s';

CREATE TABLE IF NOT EXISTS "LiteLLM_ExternalCloudAgentSource" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "vendor" TEXT NOT NULL DEFAULT 'grok_bot',
    "display_name" TEXT NOT NULL DEFAULT 'GrokBot',
    "ingest_key" TEXT NOT NULL,
    "tenant_id" TEXT,
    "admin_api_key" TEXT,
    "last_event_at" TIMESTAMP(3),
    "last_spend_sync_at" TIMESTAMP(3),
    "last_spend_sync_error" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS "LiteLLM_ExternalCloudAgent" (
    "source_id" TEXT NOT NULL REFERENCES "LiteLLM_ExternalCloudAgentSource"("id") ON DELETE CASCADE ON UPDATE CASCADE,
    "external_id" TEXT NOT NULL,
    "first_seen_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "last_seen_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "request_count" BIGINT NOT NULL DEFAULT 1,
    "last_actor_id" TEXT,
    "last_model" TEXT,
    "tools" TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    PRIMARY KEY ("source_id", "external_id")
);

CREATE INDEX IF NOT EXISTS "LiteLLM_ExternalCloudAgent_source_id_last_seen_at_idx"
    ON "LiteLLM_ExternalCloudAgent"("source_id", "last_seen_at");

CREATE TABLE IF NOT EXISTS "LiteLLM_ExternalCloudAgentSpend" (
    "source_id" TEXT NOT NULL REFERENCES "LiteLLM_ExternalCloudAgentSource"("id") ON DELETE CASCADE ON UPDATE CASCADE,
    "vendor_user_id" TEXT NOT NULL,
    "email" TEXT,
    "display_name" TEXT,
    "day" DATE NOT NULL,
    "cost_micros" BIGINT NOT NULL,
    "prompt_tokens" BIGINT NOT NULL,
    "completion_tokens" BIGINT NOT NULL,
    "requests" BIGINT NOT NULL,
    PRIMARY KEY ("source_id", "vendor_user_id", "day")
);

CREATE INDEX IF NOT EXISTS "LiteLLM_ExternalCloudAgentSpend_source_id_day_idx"
    ON "LiteLLM_ExternalCloudAgentSpend"("source_id", "day");
