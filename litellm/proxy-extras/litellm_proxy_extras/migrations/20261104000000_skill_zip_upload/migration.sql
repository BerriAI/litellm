SET LOCAL lock_timeout = '10s';

CREATE TABLE IF NOT EXISTS "LiteLLM_ClaudeCodePluginUpload" (
    "plugin_name" TEXT NOT NULL,
    "content" BYTEA NOT NULL,
    CONSTRAINT "LiteLLM_ClaudeCodePluginUpload_pkey" PRIMARY KEY ("plugin_name"),
    CONSTRAINT "LiteLLM_ClaudeCodePluginUpload_plugin_name_fkey" FOREIGN KEY ("plugin_name")
        REFERENCES "LiteLLM_ClaudeCodePluginTable"("name") ON DELETE CASCADE ON UPDATE CASCADE
);
