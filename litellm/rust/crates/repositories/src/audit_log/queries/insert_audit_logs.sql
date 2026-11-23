INSERT INTO "LiteLLM_AuditLog" (
    id,
    updated_at,
    changed_by,
    changed_by_api_key,
    action,
    table_name,
    object_id,
    before_value,
    updated_values
)
SELECT log.id,
       COALESCE(log.updated_at, NOW() AT TIME ZONE 'UTC'),
       COALESCE(log.changed_by, ''),
       COALESCE(log.changed_by_api_key, ''),
       log.action,
       log.table_name,
       log.object_id,
       log.before_value,
       log.updated_values
FROM jsonb_to_recordset($1) AS log(
    id text,
    updated_at timestamp,
    changed_by text,
    changed_by_api_key text,
    action text,
    table_name text,
    object_id text,
    before_value jsonb,
    updated_values jsonb
)
