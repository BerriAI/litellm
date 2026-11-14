SET LOCAL lock_timeout = '10s';

CREATE TABLE IF NOT EXISTS "LiteLLM_SkillSubmission" (
    "id" TEXT NOT NULL,
    "plugin_name" TEXT NOT NULL,
    "operation" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'pending_review',
    "proposal_json" TEXT NOT NULL,
    "archive_content" BYTEA,
    "submitted_by" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "base_plugin_id" TEXT,
    "base_updated_at" TIMESTAMP(3),
    "reviewed_by" TEXT,
    "reviewed_at" TIMESTAMP(3),
    "rejection_reason" TEXT,
    CONSTRAINT "LiteLLM_SkillSubmission_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "LiteLLM_SkillSubmission_submitted_by_created_at_id_idx"
    ON "LiteLLM_SkillSubmission"("submitted_by", "created_at" DESC, "id" DESC);
CREATE INDEX IF NOT EXISTS "LiteLLM_SkillSubmission_submitted_by_status_created_at_id_idx"
    ON "LiteLLM_SkillSubmission"("submitted_by", "status", "created_at" DESC, "id" DESC);
CREATE INDEX IF NOT EXISTS "LiteLLM_SkillSubmission_status_created_at_id_idx"
    ON "LiteLLM_SkillSubmission"("status", "created_at" DESC, "id" DESC);
CREATE INDEX IF NOT EXISTS "LiteLLM_SkillSubmission_created_at_id_idx"
    ON "LiteLLM_SkillSubmission"("created_at" DESC, "id" DESC);
