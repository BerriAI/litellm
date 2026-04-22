-- Add LiteLLM_EvalsTable and LiteLLM_AgentEvalsTable for Evals [Beta] feature

CREATE TABLE IF NOT EXISTS "LiteLLM_EvalsTable" (
    "eval_id"           TEXT NOT NULL DEFAULT gen_random_uuid()::text,
    "eval_name"         TEXT NOT NULL,
    "version"           INTEGER NOT NULL DEFAULT 1,
    "criteria"          JSONB NOT NULL DEFAULT '[]',
    "judge_model"       TEXT NOT NULL DEFAULT '',
    "description"       TEXT,
    "overall_threshold" DOUBLE PRECISION,
    "max_iterations"    INTEGER NOT NULL DEFAULT 1,
    "created_at"        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at"        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by"        TEXT NOT NULL DEFAULT '',
    "updated_by"        TEXT NOT NULL DEFAULT '',

    CONSTRAINT "LiteLLM_EvalsTable_pkey" PRIMARY KEY ("eval_id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_EvalsTable_eval_name_key" ON "LiteLLM_EvalsTable"("eval_name");
CREATE INDEX IF NOT EXISTS "LiteLLM_EvalsTable_eval_name_idx" ON "LiteLLM_EvalsTable"("eval_name");

CREATE TABLE IF NOT EXISTS "LiteLLM_AgentEvalsTable" (
    "agent_id"                    TEXT NOT NULL,
    "eval_id"                     TEXT NOT NULL,
    "eval_name"                   TEXT NOT NULL,
    "on_failure"                  TEXT NOT NULL DEFAULT 'block',
    "overall_threshold_override"  DOUBLE PRECISION,
    "created_at"                  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by"                  TEXT NOT NULL DEFAULT '',

    CONSTRAINT "LiteLLM_AgentEvalsTable_pkey" PRIMARY KEY ("agent_id", "eval_id")
);

CREATE INDEX IF NOT EXISTS "LiteLLM_AgentEvalsTable_agent_id_idx" ON "LiteLLM_AgentEvalsTable"("agent_id");
CREATE INDEX IF NOT EXISTS "LiteLLM_AgentEvalsTable_eval_id_idx" ON "LiteLLM_AgentEvalsTable"("eval_id");

ALTER TABLE "LiteLLM_AgentEvalsTable"
    ADD CONSTRAINT "LiteLLM_AgentEvalsTable_eval_id_fkey"
    FOREIGN KEY ("eval_id")
    REFERENCES "LiteLLM_EvalsTable"("eval_id")
    ON DELETE CASCADE ON UPDATE CASCADE
    DEFERRABLE INITIALLY DEFERRED;
