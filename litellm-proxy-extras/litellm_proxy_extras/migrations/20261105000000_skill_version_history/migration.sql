SET LOCAL lock_timeout = '10s';
CREATE TABLE IF NOT EXISTS "LiteLLM_ClaudeCodePluginVersion" (
    "id" TEXT NOT NULL,
    "plugin_id" TEXT NOT NULL,
    "version" TEXT,
    "description" TEXT,
    "manifest_json" TEXT,
    "published_at" TIMESTAMP(3),
    "published_by" TEXT,
    "archive_available" BOOLEAN NOT NULL DEFAULT false,
    "content" BYTEA,
    CONSTRAINT "LiteLLM_ClaudeCodePluginVersion_pkey" PRIMARY KEY ("id"),
    CONSTRAINT "LiteLLM_ClaudeCodePluginVersion_plugin_id_fkey" FOREIGN KEY ("plugin_id") REFERENCES "LiteLLM_ClaudeCodePluginTable"("id") ON DELETE CASCADE ON UPDATE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_ClaudeCodePluginVersion_plugin_id_version_key" ON "LiteLLM_ClaudeCodePluginVersion"("plugin_id", "version");
CREATE INDEX IF NOT EXISTS "LiteLLM_ClaudeCodePluginVersion_plugin_id_published_at_idx" ON "LiteLLM_ClaudeCodePluginVersion"("plugin_id", "published_at");
