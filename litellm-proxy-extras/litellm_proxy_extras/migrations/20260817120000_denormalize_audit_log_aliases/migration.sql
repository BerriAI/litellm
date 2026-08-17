-- AlterTable
-- Nullable columns only: a metadata-only ALTER that stays fast regardless of table size.
-- Backfill of pre-existing rows is a manual, optional operation:
-- db_scripts/backfill_audit_log_aliases.sql. New rows are stamped at write time,
-- historical rows stay NULL until an operator runs the script.
ALTER TABLE "LiteLLM_AuditLog" ADD COLUMN IF NOT EXISTS "object_alias" TEXT,
ADD COLUMN IF NOT EXISTS "object_team_id" TEXT,
ADD COLUMN IF NOT EXISTS "object_team_alias" TEXT,
ADD COLUMN IF NOT EXISTS "changed_by_user_email" TEXT,
ADD COLUMN IF NOT EXISTS "changed_by_key_alias" TEXT;
