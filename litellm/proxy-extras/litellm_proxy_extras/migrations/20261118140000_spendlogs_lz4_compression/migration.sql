SET LOCAL lock_timeout = '10s';

DO $migration$
DECLARE
    target regclass;
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_settings
        WHERE name = 'default_toast_compression' AND 'lz4' = ANY(enumvals)
    ) THEN
        RETURN;
    END IF;

    LOCK TABLE ONLY "LiteLLM_SpendLogs" IN ACCESS EXCLUSIVE MODE;
    FOR target IN
        SELECT '"LiteLLM_SpendLogs"'::regclass
        UNION ALL
        SELECT part.relid
        FROM pg_catalog.pg_partition_tree('"LiteLLM_SpendLogs"'::regclass) AS part
        JOIN pg_catalog.pg_class AS relation ON relation.oid = part.relid
        WHERE part.level > 0 AND relation.relkind IN ('p', 'r')
    LOOP
        EXECUTE format($ddl$
            ALTER TABLE ONLY %s
                ALTER COLUMN metadata SET COMPRESSION lz4,
                ALTER COLUMN messages SET COMPRESSION lz4,
                ALTER COLUMN response SET COMPRESSION lz4,
                ALTER COLUMN proxy_server_request SET COMPRESSION lz4
        $ddl$, target);
    END LOOP;
END;
$migration$;
