CREATE TABLE IF NOT EXISTS "LiteLLM_Issue" (
  "id" TEXT NOT NULL,
  "cohort_key" TEXT NOT NULL,
  "kind" TEXT NOT NULL,
  "family" TEXT NOT NULL,
  "scope_type" TEXT NOT NULL,
  "scope_id" TEXT,
  "status" TEXT NOT NULL,
  "active" BOOLEAN NOT NULL DEFAULT true,
  "ack_key" TEXT,
  "record" JSONB NOT NULL,
  "first_seen" TIMESTAMP(3) NOT NULL,
  "last_seen" TIMESTAMP(3) NOT NULL,
  "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "LiteLLM_Issue_pkey" PRIMARY KEY ("id")
);
CREATE INDEX IF NOT EXISTS "LiteLLM_Issue_status_last_seen_id_idx" ON "LiteLLM_Issue"("status", "last_seen", "id");
CREATE INDEX IF NOT EXISTS "LiteLLM_Issue_kind_family_last_seen_id_idx" ON "LiteLLM_Issue"("kind", "family", "last_seen", "id");
CREATE INDEX IF NOT EXISTS "LiteLLM_Issue_scope_type_scope_id_last_seen_id_idx" ON "LiteLLM_Issue"("scope_type", "scope_id", "last_seen", "id");
CREATE INDEX IF NOT EXISTS "LiteLLM_Issue_cohort_key_active_idx" ON "LiteLLM_Issue"("cohort_key", "active");
CREATE TABLE IF NOT EXISTS "LiteLLM_IssueSignal" (
  "issue_id" TEXT NOT NULL,
  "source" TEXT NOT NULL,
  "source_id" TEXT NOT NULL,
  "record" JSONB NOT NULL,
  CONSTRAINT "LiteLLM_IssueSignal_pkey" PRIMARY KEY ("issue_id", "source", "source_id")
);
CREATE INDEX IF NOT EXISTS "LiteLLM_IssueSignal_source_source_id_idx" ON "LiteLLM_IssueSignal"("source", "source_id");
CREATE TABLE IF NOT EXISTS "LiteLLM_IssueDelivery" (
  "id" TEXT NOT NULL,
  "issue_id" TEXT NOT NULL,
  "record" JSONB NOT NULL,
  "sent" BOOLEAN NOT NULL DEFAULT false,
  "next_attempt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "LiteLLM_IssueDelivery_pkey" PRIMARY KEY ("id")
);
CREATE INDEX IF NOT EXISTS "LiteLLM_IssueDelivery_sent_next_attempt_id_idx" ON "LiteLLM_IssueDelivery"("sent", "next_attempt", "id");
CREATE INDEX IF NOT EXISTS "LiteLLM_IssueDelivery_issue_id_idx" ON "LiteLLM_IssueDelivery"("issue_id");
CREATE TABLE IF NOT EXISTS "LiteLLM_IssueDeliveryRule" (
  "delivery_id" TEXT NOT NULL,
  "rule_id" TEXT NOT NULL,
  "source_id" TEXT NOT NULL,
  "rule_revision" TEXT NOT NULL,
  CONSTRAINT "LiteLLM_IssueDeliveryRule_pkey" PRIMARY KEY ("delivery_id", "rule_id", "source_id")
);
ALTER TABLE "LiteLLM_UsageAlertIncident" ADD COLUMN IF NOT EXISTS "evaluation_revision" TEXT NOT NULL DEFAULT '';
CREATE INDEX IF NOT EXISTS "LiteLLM_UsageAlertIncident_evaluation_day_breached_id_idx"
  ON "LiteLLM_UsageAlertIncident"("evaluation_day", "breached", "id");
