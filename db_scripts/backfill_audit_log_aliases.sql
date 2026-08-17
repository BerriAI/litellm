-- One-shot backfill of the denormalized alias columns on LiteLLM_AuditLog
-- (object_alias, object_team_id, object_team_alias, changed_by_user_email,
-- changed_by_key_alias) for rows written before the columns existed.
--
-- This is an opt-in, manual operation. New deployments do not need it: the
-- audit writer stamps the columns at write time from the moment the release
-- is deployed. Run it only if you want /audit responses and the object_team
-- filter to cover history from before the deploy.
--
-- Every statement only touches rows where the target column is NULL, so the
-- script is idempotent and safe to re-run (including after a partial run).
--
-- On a large audit table, run this in batches instead of one shot: the first
-- object_alias statement and the object_team_id statement rewrite every
-- matching row, so wrap each UPDATE with an id-range or updated_at-range
-- predicate and loop until no rows change. Run VACUUM (ANALYZE)
-- "LiteLLM_AuditLog" afterward to reclaim the dead tuples the rewrites leave
-- behind.
--
-- Sources, matching what the writer produces:
-- - object_alias comes from the before/updated JSON blobs captured at change
--   time (updated_values wins over before_value; users prefer user_alias then
--   user_email). Key rows are the exception: the writer masks key_alias inside
--   the blobs, so their alias resolves via join on the current key table and
--   rows for already-deleted keys stay NULL.
-- - The actor columns resolve via joins on the current user and key tables,
--   so deleted actors stay NULL.

UPDATE "LiteLLM_AuditLog"
SET "object_alias" = CASE "table_name"
    WHEN 'LiteLLM_TeamTable' THEN COALESCE(
        NULLIF("updated_values"->>'team_alias', ''),
        NULLIF("before_value"->>'team_alias', '')
    )
    WHEN 'LiteLLM_UserTable' THEN COALESCE(
        NULLIF("updated_values"->>'user_alias', ''),
        NULLIF("updated_values"->>'user_email', ''),
        NULLIF("before_value"->>'user_alias', ''),
        NULLIF("before_value"->>'user_email', '')
    )
    WHEN 'LiteLLM_ProxyModelTable' THEN COALESCE(
        NULLIF("updated_values"->>'model_name', ''),
        NULLIF("before_value"->>'model_name', '')
    )
END
WHERE "object_alias" IS NULL
  AND "table_name" IN ('LiteLLM_TeamTable', 'LiteLLM_UserTable', 'LiteLLM_ProxyModelTable');

UPDATE "LiteLLM_AuditLog" a
SET "object_alias" = v."key_alias"
FROM "LiteLLM_VerificationToken" v
WHERE a."object_alias" IS NULL
  AND a."table_name" = 'LiteLLM_VerificationToken'
  AND a."object_id" = v."token"
  AND v."key_alias" IS NOT NULL
  AND v."key_alias" <> '';

UPDATE "LiteLLM_AuditLog"
SET "object_team_id" = COALESCE(
    NULLIF("updated_values"->>'team_id', ''),
    NULLIF("before_value"->>'team_id', '')
)
WHERE "object_team_id" IS NULL;

UPDATE "LiteLLM_AuditLog"
SET "object_team_id" = "object_id"
WHERE "object_team_id" IS NULL
  AND "table_name" = 'LiteLLM_TeamTable';

UPDATE "LiteLLM_AuditLog"
SET "object_team_alias" = COALESCE(
    NULLIF("updated_values"->>'team_alias', ''),
    NULLIF("before_value"->>'team_alias', '')
)
WHERE "object_team_alias" IS NULL;

UPDATE "LiteLLM_AuditLog" a
SET "object_team_alias" = t."team_alias"
FROM "LiteLLM_TeamTable" t
WHERE a."object_team_alias" IS NULL
  AND a."object_team_id" = t."team_id"
  AND t."team_alias" IS NOT NULL
  AND t."team_alias" <> '';

UPDATE "LiteLLM_AuditLog" a
SET "changed_by_user_email" = u."user_email"
FROM "LiteLLM_UserTable" u
WHERE a."changed_by_user_email" IS NULL
  AND a."changed_by" = u."user_id"
  AND u."user_email" IS NOT NULL
  AND u."user_email" <> '';

UPDATE "LiteLLM_AuditLog" a
SET "changed_by_key_alias" = v."key_alias"
FROM "LiteLLM_VerificationToken" v
WHERE a."changed_by_key_alias" IS NULL
  AND a."changed_by_api_key" = v."token"
  AND v."key_alias" IS NOT NULL
  AND v."key_alias" <> '';
