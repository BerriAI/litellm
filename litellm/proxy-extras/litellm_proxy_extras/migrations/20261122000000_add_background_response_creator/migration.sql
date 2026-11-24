SET LOCAL lock_timeout = '10s';

-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_BackgroundResponseCreator" (
    "response_id" TEXT NOT NULL,
    "upstream_url" TEXT NOT NULL,
    "create_context" JSONB NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "poll_lease_expires_at" TIMESTAMP(3),

    CONSTRAINT "LiteLLM_BackgroundResponseCreator_pkey" PRIMARY KEY ("response_id")
);

-- CreateIndex
CREATE INDEX IF NOT EXISTS "idx_background_response_creator_created_at" ON "LiteLLM_BackgroundResponseCreator"("created_at");
