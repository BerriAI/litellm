-- One row per (router, request_type, model). Hot path on every routing decision.
CREATE TABLE "LiteLLM_AdaptiveRouterState" (
    router_name      TEXT NOT NULL,
    request_type     TEXT NOT NULL,
    model_name       TEXT NOT NULL,
    alpha            DOUBLE PRECISION NOT NULL,
    beta             DOUBLE PRECISION NOT NULL,
    total_samples    INTEGER NOT NULL DEFAULT 0,
    last_updated_at  TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (router_name, request_type, model_name)
);

-- One row per (session, router, model). Updated per turn via the queue.
CREATE TABLE "LiteLLM_AdaptiveRouterSession" (
    session_id              TEXT NOT NULL,
    router_name             TEXT NOT NULL,
    model_name              TEXT NOT NULL,
    classified_type         TEXT NOT NULL,
    misalignment_count      INTEGER NOT NULL DEFAULT 0,
    stagnation_count        INTEGER NOT NULL DEFAULT 0,
    disengagement_count     INTEGER NOT NULL DEFAULT 0,
    satisfaction_count      INTEGER NOT NULL DEFAULT 0,
    failure_count           INTEGER NOT NULL DEFAULT 0,
    loop_count              INTEGER NOT NULL DEFAULT 0,
    exhaustion_count        INTEGER NOT NULL DEFAULT 0,
    last_user_content       TEXT,
    last_assistant_content  TEXT,
    tool_call_history       JSONB NOT NULL DEFAULT '[]',
    pending_tool_calls      JSONB NOT NULL DEFAULT '{}',
    turn_count              INTEGER NOT NULL DEFAULT 0,
    last_processed_turn     INTEGER NOT NULL DEFAULT -1,
    clean_credit_awarded    BOOLEAN NOT NULL DEFAULT FALSE,
    terminal_status         INTEGER,
    last_activity_at        TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, router_name, model_name)
);

CREATE INDEX "idx_adaptive_router_session_activity"
    ON "LiteLLM_AdaptiveRouterSession" (last_activity_at);
