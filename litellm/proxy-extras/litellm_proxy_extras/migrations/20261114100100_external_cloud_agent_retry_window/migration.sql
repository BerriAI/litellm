SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_ExternalCloudAgent"
    ADD COLUMN IF NOT EXISTS "recent_request_ids" TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[];
