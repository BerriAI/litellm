SET LOCAL lock_timeout = '10s';

CREATE TABLE IF NOT EXISTS "LiteLLM_UsageAlertRule" (
 "id" TEXT PRIMARY KEY, "scope_type" TEXT NOT NULL, "scope_id" TEXT,
 "enabled" BOOLEAN NOT NULL DEFAULT true, "record" JSONB NOT NULL,
 "deleted" BOOLEAN NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS "LiteLLM_UsageAlertRule_enabled_deleted_id_idx" ON "LiteLLM_UsageAlertRule" ("enabled", "deleted", "id");
CREATE INDEX IF NOT EXISTS "LiteLLM_UsageAlertRule_scope_type_scope_id_deleted_idx" ON "LiteLLM_UsageAlertRule" ("scope_type", "scope_id", "deleted");
CREATE TABLE IF NOT EXISTS "LiteLLM_UsageAlertIncident" (
 "id" TEXT PRIMARY KEY, "rule_id" TEXT NOT NULL, "scope_key" TEXT NOT NULL,
 "scope_type" TEXT NOT NULL, "scope_id" TEXT, "metric" TEXT NOT NULL,
 "status" TEXT NOT NULL, "day" TEXT NOT NULL, "evaluation_day" TEXT NOT NULL,
 "breached" BOOLEAN NOT NULL, "record" JSONB NOT NULL,
 "notified_channels" TEXT[] DEFAULT ARRAY[]::TEXT[],
 "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "LiteLLM_UsageAlertIncident_rule_id_scope_key_evaluation_day_idx" ON "LiteLLM_UsageAlertIncident" ("rule_id", "scope_key", "evaluation_day");
CREATE INDEX IF NOT EXISTS "LiteLLM_UsageAlertIncident_scope_type_scope_id_day_idx" ON "LiteLLM_UsageAlertIncident" ("scope_type", "scope_id", "day");
CREATE INDEX IF NOT EXISTS "LiteLLM_UsageAlertIncident_status_day_idx" ON "LiteLLM_UsageAlertIncident" ("status", "day");
CREATE INDEX IF NOT EXISTS "LiteLLM_UsageAlertIncident_updated_at_idx" ON "LiteLLM_UsageAlertIncident" ("updated_at");
CREATE TABLE IF NOT EXISTS "LiteLLM_UsageAlertCheckpoint" (
 "id" TEXT PRIMARY KEY, "completed" BOOLEAN NOT NULL DEFAULT false,
 "owner" TEXT, "expires_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "LiteLLM_UsageAlertCheckpoint_expires_at_idx" ON "LiteLLM_UsageAlertCheckpoint" ("expires_at");
