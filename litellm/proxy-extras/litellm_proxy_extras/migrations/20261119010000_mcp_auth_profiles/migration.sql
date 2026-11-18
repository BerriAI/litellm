SET LOCAL lock_timeout = '10s';

CREATE TABLE IF NOT EXISTS "LiteLLM_MCPAuthProfileTable" (
    "profile_id" TEXT NOT NULL,
    "profile_id_normalized" TEXT NOT NULL,
    "server_id" TEXT NOT NULL,
    "auth_type" TEXT NOT NULL,
    "oauth2_flow" TEXT,
    "credentials" JSONB NOT NULL DEFAULT '{}',
    "token_url" TEXT,
    "scopes" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT,

    CONSTRAINT "LiteLLM_MCPAuthProfileTable_pkey" PRIMARY KEY ("profile_id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_MCPAuthProfileTable_profile_id_normalized_key" ON "LiteLLM_MCPAuthProfileTable"("profile_id_normalized");

CREATE INDEX IF NOT EXISTS "LiteLLM_MCPAuthProfileTable_server_id_idx" ON "LiteLLM_MCPAuthProfileTable"("server_id");

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_MCPAuthProfileTable_server_id_fkey') THEN
        ALTER TABLE "LiteLLM_MCPAuthProfileTable" ADD CONSTRAINT "LiteLLM_MCPAuthProfileTable_server_id_fkey" FOREIGN KEY ("server_id") REFERENCES "LiteLLM_MCPServerTable"("server_id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;
