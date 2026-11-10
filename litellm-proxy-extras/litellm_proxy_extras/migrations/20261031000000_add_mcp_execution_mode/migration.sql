SET LOCAL lock_timeout = '10s';

DO $$
BEGIN
    CREATE TYPE "MCPExecutionMode" AS ENUM ('litellm', 'provider');
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE "LiteLLM_MCPServerTable"
ADD COLUMN IF NOT EXISTS "execution_mode" "MCPExecutionMode" NOT NULL DEFAULT 'litellm',
ALTER COLUMN "transport" DROP NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_MCPServerTable_provider_url_key"
ON "LiteLLM_MCPServerTable" ("url")
WHERE "execution_mode" = 'provider';
