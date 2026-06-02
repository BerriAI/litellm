-- Cloud Agents settings (Epic G / LIT-2891).
--
-- Adds four tables that back the new Settings -> Cloud Agents UI:
--   * LiteLLM_AgentVMConfig            -- per-team provider, AWS BYOC, warm pool, network access
--   * LiteLLM_AgentSecret              -- per-team encrypted secrets, write-only on read
--   * LiteLLM_AgentWorker              -- self-hosted worker registrations
--   * LiteLLM_AgentWorkerPairingToken  -- single-use 15-min pairing tokens
--
-- Encryption uses the existing nacl/SecretBox path in
-- litellm.proxy.common_utils.encrypt_decrypt_utils — no new KMS infra here.
-- Values are stored base64-encoded; raw secrets are NEVER returned from any
-- GET endpoint (LIT-2891 validation #2).

CREATE TABLE IF NOT EXISTS "LiteLLM_AgentVMConfig" (
    "team_id"                   TEXT PRIMARY KEY,
    "provider"                  TEXT NOT NULL DEFAULT 'disabled',
    "aws_auth_method"           TEXT,
    "aws_access_key_id_enc"     TEXT,
    "aws_secret_access_key_enc" TEXT,
    "aws_role_arn_enc"          TEXT,
    "aws_region"                TEXT,
    "ami_id"                    TEXT,
    "instance_type"             TEXT,
    "subnet_id"                 TEXT,
    "security_group_id"         TEXT,
    "iam_instance_profile"      TEXT,
    "use_spot"                  BOOLEAN NOT NULL DEFAULT TRUE,
    "max_session_minutes"       INTEGER NOT NULL DEFAULT 120,
    "warm_pool_enabled"         BOOLEAN NOT NULL DEFAULT FALSE,
    "warm_pool_size"            INTEGER NOT NULL DEFAULT 0,
    "max_idle_minutes"          INTEGER NOT NULL DEFAULT 30,
    "hydrate_transport"         TEXT NOT NULL DEFAULT 'auto',
    "network_access"            JSONB NOT NULL DEFAULT '{"mode":"allow_all","allowlist":[]}'::jsonb,
    "self_hosted_enabled"       BOOLEAN NOT NULL DEFAULT FALSE,
    "created_at"                TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at"                TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS "LiteLLM_AgentSecret" (
    "id"         TEXT PRIMARY KEY,
    "team_id"    TEXT NOT NULL,
    "name"       TEXT NOT NULL,
    "value_enc"  TEXT NOT NULL,
    "scope"      JSONB NOT NULL DEFAULT '"all"'::jsonb,
    "type"       TEXT NOT NULL DEFAULT 'env',
    "file_path"  TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_AgentSecret_team_id_name_key"
    ON "LiteLLM_AgentSecret" ("team_id", "name");
CREATE INDEX IF NOT EXISTS "LiteLLM_AgentSecret_team_id_idx"
    ON "LiteLLM_AgentSecret" ("team_id");

CREATE TABLE IF NOT EXISTS "LiteLLM_AgentWorker" (
    "id"               TEXT PRIMARY KEY,
    "team_id"          TEXT NOT NULL,
    "hostname"         TEXT NOT NULL,
    "status"           TEXT NOT NULL DEFAULT 'offline',
    "last_seen_at"     TIMESTAMP(3),
    "cpu_pct"          DOUBLE PRECISION,
    "mem_gb"           DOUBLE PRECISION,
    "active_sessions"  INTEGER NOT NULL DEFAULT 0,
    "worker_jwt_hash"  TEXT NOT NULL,
    "created_at"       TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "LiteLLM_AgentWorker_team_id_status_idx"
    ON "LiteLLM_AgentWorker" ("team_id", "status");
-- Lookup by JWT digest is on the hot path — every worker long-poll
-- heartbeat (LIT-2890 / B2) calls `find_worker_by_jwt`, which filters
-- by this column. Without an index that's a full scan per heartbeat.
CREATE INDEX IF NOT EXISTS "LiteLLM_AgentWorker_worker_jwt_hash_idx"
    ON "LiteLLM_AgentWorker" ("worker_jwt_hash");

CREATE TABLE IF NOT EXISTS "LiteLLM_AgentWorkerPairingToken" (
    "token_hash" TEXT PRIMARY KEY,
    "team_id"    TEXT NOT NULL,
    "created_by" TEXT NOT NULL,
    "expires_at" TIMESTAMP(3) NOT NULL,
    "used_at"    TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "LiteLLM_AgentWorkerPairingToken_team_id_idx"
    ON "LiteLLM_AgentWorkerPairingToken" ("team_id");
