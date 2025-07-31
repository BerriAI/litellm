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

-- Migration for existing tables: rename alias to server_name if upgrading
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'LiteLLM_MCPServerTable' AND column_name = 'alias') THEN
        ALTER TABLE "LiteLLM_MCPServerTable" RENAME COLUMN "alias" TO "server_name";
    END IF;
END $$;
-- Migration for existing tables: add alias column if upgrading
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN IF NOT EXISTS "alias" TEXT;

