CREATE TABLE IF NOT EXISTS "LiteLLM_MCPToolVersion" (
    "id" TEXT NOT NULL,
    "server_id" TEXT NOT NULL,
    "tool_name" TEXT NOT NULL,
    "version" INTEGER NOT NULL,
    "description" TEXT NOT NULL DEFAULT '',
    "input_schema" JSONB NOT NULL DEFAULT '{}',
    "change_kind" TEXT NOT NULL,
    "changes" JSONB NOT NULL DEFAULT '[]',
    "changelog" TEXT,
    "deprecated_at" TIMESTAMP(3),
    "sunset_date" TIMESTAMP(3),
    "deprecation_note" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,

    CONSTRAINT "LiteLLM_MCPToolVersion_pkey" PRIMARY KEY ("id")
);
CREATE INDEX IF NOT EXISTS "LiteLLM_MCPToolVersion_server_id_idx" ON "LiteLLM_MCPToolVersion"("server_id");
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_MCPToolVersion_server_id_tool_name_version_key"
ON "LiteLLM_MCPToolVersion"("server_id", "tool_name", "version");
