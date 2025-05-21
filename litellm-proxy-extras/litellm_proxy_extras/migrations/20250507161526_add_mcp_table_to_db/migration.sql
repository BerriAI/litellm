-- CreateTable
CREATE TABLE "LiteLLM_MCPServerTable" (
    "server_id" TEXT NOT NULL,
    "alias" TEXT,
    "description" TEXT,
    "url" TEXT NOT NULL,
    "transport" TEXT NOT NULL DEFAULT 'sse',
    "spec_version" TEXT NOT NULL DEFAULT '2025-03-26',
    "auth_type" TEXT,
    "created_at" TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT,

    CONSTRAINT "LiteLLM_MCPServerTable_pkey" PRIMARY KEY ("server_id")
);

