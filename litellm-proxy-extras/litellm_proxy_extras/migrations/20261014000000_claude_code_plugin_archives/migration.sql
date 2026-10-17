SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_ClaudeCodePluginTable" ADD COLUMN IF NOT EXISTS "archive_sha256" TEXT;

CREATE TABLE IF NOT EXISTS "LiteLLM_ClaudeCodePluginArchiveTable" (
    "plugin_id" TEXT NOT NULL,
    "content" BYTEA NOT NULL,
    "size_bytes" INTEGER NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_ClaudeCodePluginArchiveTable_pkey" PRIMARY KEY ("plugin_id")
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'LiteLLM_ClaudeCodePluginArchiveTable_plugin_id_fkey'
            AND conrelid = '"LiteLLM_ClaudeCodePluginArchiveTable"'::regclass
    ) THEN
        ALTER TABLE "LiteLLM_ClaudeCodePluginArchiveTable"
        ADD CONSTRAINT "LiteLLM_ClaudeCodePluginArchiveTable_plugin_id_fkey"
        FOREIGN KEY ("plugin_id") REFERENCES "LiteLLM_ClaudeCodePluginTable"("id")
        ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;
