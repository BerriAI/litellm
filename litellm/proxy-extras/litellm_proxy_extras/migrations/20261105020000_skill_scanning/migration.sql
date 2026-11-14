SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_SkillSubmission" ADD COLUMN IF NOT EXISTS "scan_json" TEXT;
