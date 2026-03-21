-- Ensure project_id column exists in LiteLLM_VerificationToken.
-- The original migration (20251113000000_add_project_table) adds this column,
-- but if it failed partway through (e.g. LiteLLM_ProjectTable already existed)
-- and was resolved as idempotent, the ALTER TABLE step may have been skipped.
ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN IF NOT EXISTS "project_id" TEXT;
