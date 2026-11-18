DO $migration$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_catalog.pg_settings
        WHERE name = 'default_toast_compression' AND 'lz4' = ANY(enumvals)
    ) THEN
        PERFORM pg_catalog.set_config('lock_timeout', '10s', true);
        EXECUTE $ddl$
            ALTER TABLE ONLY "LiteLLM_SpendLogs"
                ALTER COLUMN metadata SET COMPRESSION lz4,
                ALTER COLUMN messages SET COMPRESSION lz4,
                ALTER COLUMN response SET COMPRESSION lz4,
                ALTER COLUMN proxy_server_request SET COMPRESSION lz4
        $ddl$;
    END IF;
END;
$migration$;
