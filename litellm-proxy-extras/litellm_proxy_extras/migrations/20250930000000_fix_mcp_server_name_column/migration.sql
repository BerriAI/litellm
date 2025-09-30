-- Emergency fix for MCP server_name column issue
-- This migration ensures the server_name column exists regardless of previous migration state
-- Safe to run multiple times - it's idempotent

DO $$
BEGIN
    -- Check if table exists first
    IF EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'LiteLLM_MCPServerTable'
    ) THEN
        -- If server_name doesn't exist but alias does, rename it
        IF NOT EXISTS (
            SELECT FROM information_schema.columns 
            WHERE table_name = 'LiteLLM_MCPServerTable' 
            AND column_name = 'server_name'
        ) AND EXISTS (
            SELECT FROM information_schema.columns 
            WHERE table_name = 'LiteLLM_MCPServerTable' 
            AND column_name = 'alias'
        ) THEN
            RAISE NOTICE 'Renaming alias to server_name...';
            ALTER TABLE "LiteLLM_MCPServerTable" RENAME COLUMN "alias" TO "server_name";
        END IF;

        -- Ensure server_name column exists (in case it was never there)
        IF NOT EXISTS (
            SELECT FROM information_schema.columns 
            WHERE table_name = 'LiteLLM_MCPServerTable' 
            AND column_name = 'server_name'
        ) THEN
            RAISE NOTICE 'Adding server_name column...';
            ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN "server_name" TEXT;
        END IF;

        -- Ensure alias column exists (for backward compatibility)
        IF NOT EXISTS (
            SELECT FROM information_schema.columns 
            WHERE table_name = 'LiteLLM_MCPServerTable' 
            AND column_name = 'alias'
        ) THEN
            RAISE NOTICE 'Adding alias column for backward compatibility...';
            ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN "alias" TEXT;
        END IF;

        RAISE NOTICE 'MCP server_name column fix completed successfully';
    ELSE
        RAISE NOTICE 'LiteLLM_MCPServerTable does not exist, skipping fix';
    END IF;
END $$;
