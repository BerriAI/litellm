CREATE TABLE IF NOT EXISTS "LiteLLM_AutoRouterSession" (
    api_key            TEXT NOT NULL,
    session_id         TEXT NOT NULL,
    model_group        TEXT NOT NULL,
    router_kind        TEXT NOT NULL,
    baseline_model     TEXT,
    turns              INTEGER NOT NULL DEFAULT 0,
    turns_with_usage   INTEGER NOT NULL DEFAULT 0,
    total_tokens       BIGINT NOT NULL DEFAULT 0,
    spend              DOUBLE PRECISION NOT NULL DEFAULT 0,
    baseline_spend     DOUBLE PRECISION NOT NULL DEFAULT 0,
    first_visit_turns  INTEGER NOT NULL DEFAULT 0,
    first_visit_hits   INTEGER NOT NULL DEFAULT 0,
    warm_turns         INTEGER NOT NULL DEFAULT 0,
    warm_hits          INTEGER NOT NULL DEFAULT 0,
    expired_turns      INTEGER NOT NULL DEFAULT 0,
    expired_hits       INTEGER NOT NULL DEFAULT 0,
    unordered_turns    INTEGER NOT NULL DEFAULT 0,
    unordered_hits     INTEGER NOT NULL DEFAULT 0,
    unknown_ttl_turns        INTEGER NOT NULL DEFAULT 0,
    unknown_ttl_hits         INTEGER NOT NULL DEFAULT 0,
    cache_5m_turns           INTEGER NOT NULL DEFAULT 0,
    cache_1h_turns           INTEGER NOT NULL DEFAULT 0,
    cache_ttl_unknown_turns INTEGER NOT NULL DEFAULT 0,
    tiers              JSONB NOT NULL DEFAULT '{}',
    first_turn_at      TIMESTAMP(3) NOT NULL,
    last_turn_at       TIMESTAMP(3) NOT NULL,
    updated_at         TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_AutoRouterSession_pkey" PRIMARY KEY (api_key, session_id, model_group)
);

CREATE INDEX IF NOT EXISTS "idx_auto_router_session_started"
    ON "LiteLLM_AutoRouterSession" (first_turn_at);

CREATE INDEX IF NOT EXISTS "idx_auto_router_session_activity"
    ON "LiteLLM_AutoRouterSession" (last_turn_at);
