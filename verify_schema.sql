-- Connect to your PostgreSQL database and run these queries to verify:

-- 1. Check if the table exists
SELECT EXISTS (
    SELECT FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_name = 'LiteLLM_MCPServerTable'
);

-- 2. Check the table structure
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns 
WHERE table_name = 'LiteLLM_MCPServerTable' 
AND table_schema = 'public'
ORDER BY ordinal_position;

-- 3. Expected columns after both migrations:
-- server_id (PRIMARY KEY)
-- server_name (TEXT, nullable) <- This should exist after migration
-- alias (TEXT, nullable) <- This should also exist
-- description, url, transport, spec_version, auth_type, created_at, created_by, updated_at, updated_by

-- 4. If you have data, check that it migrated correctly:
-- SELECT server_id, server_name, alias FROM "LiteLLM_MCPServerTable" LIMIT 5;
