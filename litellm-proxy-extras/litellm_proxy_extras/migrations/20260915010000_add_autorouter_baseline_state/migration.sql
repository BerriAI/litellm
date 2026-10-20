CREATE TABLE IF NOT EXISTS "LiteLLM_AutoRouterBaselineComparison" (
    "scope" TEXT PRIMARY KEY,
    "api_key" TEXT NOT NULL,
    "session_id" TEXT NOT NULL,
    "router_name" TEXT NOT NULL,
    "initial_equivalent" BOOLEAN NOT NULL,
    "revision" BIGINT NOT NULL DEFAULT 0,
    "published_revision" BIGINT NOT NULL DEFAULT 0,
    "history" TEXT,
    "attempted_at" TIMESTAMP(3),
    "retired" BOOLEAN NOT NULL DEFAULT FALSE,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS "idx_autorouter_baseline_scope"
    ON "LiteLLM_AutoRouterBaselineComparison" ("api_key", "session_id", "router_name");
CREATE INDEX IF NOT EXISTS "idx_autorouter_baseline_updated"
    ON "LiteLLM_AutoRouterBaselineComparison" ("updated_at");
CREATE INDEX IF NOT EXISTS "idx_autorouter_baseline_dirty"
    ON "LiteLLM_AutoRouterBaselineComparison" ("attempted_at", "updated_at", "scope")
    WHERE NOT "retired" AND "revision" <> "published_revision";

CREATE TABLE IF NOT EXISTS "LiteLLM_AutoRouterBaselineObservation" (
    "request_id" TEXT PRIMARY KEY,
    "scope" TEXT NOT NULL,
    "started_at" DOUBLE PRECISION NOT NULL,
    "revision" BIGINT NOT NULL,
    "data" TEXT NOT NULL,
    "publication" TEXT,
    "conflicted" BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS "idx_autorouter_baseline_event_order"
    ON "LiteLLM_AutoRouterBaselineObservation" ("scope", "started_at", "request_id");
CREATE INDEX IF NOT EXISTS "idx_autorouter_baseline_event_revision"
    ON "LiteLLM_AutoRouterBaselineObservation" ("scope", "revision", "started_at");
