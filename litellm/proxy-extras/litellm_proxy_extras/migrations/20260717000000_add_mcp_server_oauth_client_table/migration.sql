-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_MCPServerOAuthClient" (
    "server_id" TEXT NOT NULL,
    "credentials" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_MCPServerOAuthClient_pkey" PRIMARY KEY ("server_id")
);
