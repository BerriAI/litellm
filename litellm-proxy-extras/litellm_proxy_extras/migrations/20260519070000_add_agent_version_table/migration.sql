-- Migration: add_agent_version_table (S3-03)
--
-- Persists a snapshot of each agent's content fields on every PUT/PATCH so
-- a) operators can see what changed, and b) rollback is a single insert
-- copying an older snapshot back over the live row.
--
-- The table is append-only — no UPDATE / DELETE on rows from the proxy.
-- Rollback writes a *new* version row, never mutates an old one.

CREATE TABLE IF NOT EXISTS "LiteLLM_AgentVersionTable" (
  "version_id"        TEXT        PRIMARY KEY,
  "agent_id"          TEXT        NOT NULL,
  "version_number"    INTEGER     NOT NULL,
  "agent_card_params" JSONB       NOT NULL,
  "litellm_params"    JSONB,
  "static_headers"    JSONB,
  "created_by"        TEXT,
  "created_at"        TIMESTAMP   NOT NULL DEFAULT NOW(),
  "is_rollback"       BOOLEAN     NOT NULL DEFAULT FALSE,
  "rolled_back_from"  INTEGER,    -- version_number this row restored

  CONSTRAINT "LiteLLM_AgentVersionTable_agent_id_version_number_unique"
    UNIQUE ("agent_id", "version_number"),
  CONSTRAINT "LiteLLM_AgentVersionTable_agent_fk"
    FOREIGN KEY ("agent_id") REFERENCES "LiteLLM_AgentsTable" ("agent_id")
    ON DELETE CASCADE
);

-- History fetch hot path: GET /v1/agents/{id}/versions ORDER BY version DESC
CREATE INDEX IF NOT EXISTS "LiteLLM_AgentVersionTable_agent_id_version_idx"
  ON "LiteLLM_AgentVersionTable" ("agent_id", "version_number" DESC);
