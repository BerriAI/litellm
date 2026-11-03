CREATE TYPE "MCPExecutionMode" AS ENUM ('litellm', 'provider');

ALTER TABLE "LiteLLM_MCPServerTable"
ADD COLUMN "execution_mode" "MCPExecutionMode" NOT NULL DEFAULT 'litellm',
ALTER COLUMN "transport" DROP NOT NULL;

CREATE UNIQUE INDEX "LiteLLM_MCPServerTable_provider_url_key"
ON "LiteLLM_MCPServerTable" ("url")
WHERE "execution_mode" = 'provider';
