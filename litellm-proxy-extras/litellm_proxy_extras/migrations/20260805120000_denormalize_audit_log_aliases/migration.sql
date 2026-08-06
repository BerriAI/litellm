-- AlterTable
ALTER TABLE "LiteLLM_AuditLog" ADD COLUMN IF NOT EXISTS "object_alias" TEXT,
ADD COLUMN IF NOT EXISTS "object_team_id" TEXT,
ADD COLUMN IF NOT EXISTS "object_team_alias" TEXT,
ADD COLUMN IF NOT EXISTS "changed_by_user_email" TEXT,
ADD COLUMN IF NOT EXISTS "changed_by_key_alias" TEXT;

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_AuditLog_table_name_object_id_idx" ON "LiteLLM_AuditLog"("table_name", "object_id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_AuditLog_object_team_id_idx" ON "LiteLLM_AuditLog"("object_team_id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_AuditLog_updated_at_idx" ON "LiteLLM_AuditLog"("updated_at");

-- Backfill object_alias from the JSON blobs captured at change time (updated_values wins over before_value).
-- Key rows are excluded here: the audit writer masks key_alias inside the blobs, so they resolve via join below
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
    WHEN 'LiteLLM_OrganizationTable' THEN COALESCE(
        NULLIF("updated_values"->>'organization_alias', ''),
        NULLIF("before_value"->>'organization_alias', '')
    )
    WHEN 'LiteLLM_ProxyModelTable' THEN COALESCE(
        NULLIF("updated_values"->>'model_name', ''),
        NULLIF("before_value"->>'model_name', '')
    )
END
WHERE "object_alias" IS NULL;

-- Backfill object_alias for key rows via the current key table (blob key_alias is masked; deleted keys stay NULL)
UPDATE "LiteLLM_AuditLog" a
SET "object_alias" = v."key_alias"
FROM "LiteLLM_VerificationToken" v
WHERE a."object_alias" IS NULL
  AND a."table_name" = 'LiteLLM_VerificationToken'
  AND a."object_id" = v."token"
  AND v."key_alias" IS NOT NULL;

-- Backfill object_team_id from the JSON blobs
UPDATE "LiteLLM_AuditLog"
SET "object_team_id" = COALESCE(
    NULLIF("updated_values"->>'team_id', ''),
    NULLIF("before_value"->>'team_id', '')
)
WHERE "object_team_id" IS NULL;

-- Backfill object_team_alias from the JSON blobs (team rows carry it even after the team is deleted)
UPDATE "LiteLLM_AuditLog"
SET "object_team_alias" = COALESCE(
    NULLIF("updated_values"->>'team_alias', ''),
    NULLIF("before_value"->>'team_alias', '')
)
WHERE "object_team_alias" IS NULL;

-- Backfill object_team_alias for remaining rows via the current team table
UPDATE "LiteLLM_AuditLog" a
SET "object_team_alias" = t."team_alias"
FROM "LiteLLM_TeamTable" t
WHERE a."object_team_alias" IS NULL
  AND a."object_team_id" = t."team_id"
  AND t."team_alias" IS NOT NULL;

-- Backfill changed_by_user_email via the current user table (deleted actors stay NULL)
UPDATE "LiteLLM_AuditLog" a
SET "changed_by_user_email" = u."user_email"
FROM "LiteLLM_UserTable" u
WHERE a."changed_by_user_email" IS NULL
  AND a."changed_by" = u."user_id"
  AND u."user_email" IS NOT NULL;

-- Backfill changed_by_key_alias via the current key table (deleted keys stay NULL)
UPDATE "LiteLLM_AuditLog" a
SET "changed_by_key_alias" = v."key_alias"
FROM "LiteLLM_VerificationToken" v
WHERE a."changed_by_key_alias" IS NULL
  AND a."changed_by_api_key" = v."token"
  AND v."key_alias" IS NOT NULL;
