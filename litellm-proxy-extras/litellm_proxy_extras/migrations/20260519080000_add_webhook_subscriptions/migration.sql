-- Migration: add_webhook_subscriptions (S6-04 + groundwork for S6-05/06)
--
-- A webhook subscription = (events[], target_url, secret) belonging to an
-- app_id (S4-02 placeholder; nullable until S4 lands so we can start
-- collecting subscriptions for already-scoped keys today).
--
-- DLQ table (S6-05) lives in the same migration so the delivery service
-- has a place to park failed payloads from day one — keeps the rollout
-- atomic.

CREATE TABLE IF NOT EXISTS "LiteLLM_WebhookSubscriptionTable" (
  "subscription_id"  TEXT        PRIMARY KEY,
  "app_id"           TEXT,
  "team_id"          TEXT,
  "user_id"          TEXT,
  "events"           TEXT[]      NOT NULL DEFAULT '{}'::text[],
  "target_url"       TEXT        NOT NULL,
  "secret_hash"      TEXT        NOT NULL,
  "filters"          JSONB,
  "is_active"        BOOLEAN     NOT NULL DEFAULT TRUE,
  "created_at"       TIMESTAMP   NOT NULL DEFAULT NOW(),
  "created_by"       TEXT,
  "updated_at"       TIMESTAMP   NOT NULL DEFAULT NOW(),
  "last_success_at"  TIMESTAMP,
  "last_failure_at"  TIMESTAMP,
  "consecutive_failures" INTEGER NOT NULL DEFAULT 0
);

-- Hot path for the delivery dispatcher: "give me all active subscriptions
-- listening for this event type within this app."
CREATE INDEX IF NOT EXISTS "LiteLLM_WebhookSubscriptionTable_app_active_idx"
  ON "LiteLLM_WebhookSubscriptionTable" ("app_id", "is_active");

CREATE INDEX IF NOT EXISTS "LiteLLM_WebhookSubscriptionTable_events_gin_idx"
  ON "LiteLLM_WebhookSubscriptionTable" USING GIN ("events");


CREATE TABLE IF NOT EXISTS "LiteLLM_WebhookDLQ" (
  "dlq_id"          TEXT        PRIMARY KEY,
  "subscription_id" TEXT        NOT NULL,
  "event_type"      TEXT        NOT NULL,
  "payload"         JSONB       NOT NULL,
  "last_error"      TEXT,
  "attempts"        INTEGER     NOT NULL DEFAULT 0,
  "first_attempt_at" TIMESTAMP  NOT NULL DEFAULT NOW(),
  "last_attempt_at"  TIMESTAMP  NOT NULL DEFAULT NOW(),
  CONSTRAINT "LiteLLM_WebhookDLQ_subscription_fk"
    FOREIGN KEY ("subscription_id")
    REFERENCES "LiteLLM_WebhookSubscriptionTable" ("subscription_id")
    ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS "LiteLLM_WebhookDLQ_subscription_idx"
  ON "LiteLLM_WebhookDLQ" ("subscription_id", "last_attempt_at" DESC);
