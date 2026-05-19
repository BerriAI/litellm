-- Migration: add_xct_app_tenancy (S4-02 + groundwork for S4-05/06)
--
-- Three additive changes:
--   1. New table LiteLLM_XCTAppTable — the "app" identity
--   2. New table LiteLLM_OAuthAuthorizationCode — short-lived auth codes
--      issued by /oauth/authorize and consumed by /oauth/token
--   3. LiteLLM_VerificationToken gains app_id + token_type so OAuth-issued
--      tokens are distinguishable from legacy admin-provisioned virtual keys

CREATE TABLE IF NOT EXISTS "LiteLLM_XCTAppTable" (
  "app_id"                    TEXT       PRIMARY KEY,
  "app_name"                  TEXT       NOT NULL UNIQUE,
  "display_name"              TEXT       NOT NULL,
  "description"               TEXT,
  "icon_url"                  TEXT,
  "oauth_client_id"           TEXT       NOT NULL UNIQUE,
  "oauth_client_secret_hash"  TEXT       NOT NULL,
  "redirect_uris"             TEXT[]     NOT NULL DEFAULT '{}'::text[],
  "default_team_id"           TEXT,
  "default_scopes"            TEXT[]     NOT NULL DEFAULT '{}'::text[],
  "capability_scope_id"       TEXT,                                  -- FK to access group, soft (no constraint)
  "rpm_limit"                 INTEGER,
  "daily_budget"              NUMERIC(18, 6),
  "is_active"                 BOOLEAN    NOT NULL DEFAULT TRUE,
  "created_at"                TIMESTAMP  NOT NULL DEFAULT NOW(),
  "created_by"                TEXT,
  "updated_at"                TIMESTAMP  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS "LiteLLM_XCTAppTable_oauth_client_id_idx"
  ON "LiteLLM_XCTAppTable" ("oauth_client_id");


CREATE TABLE IF NOT EXISTS "LiteLLM_OAuthAuthorizationCode" (
  "code"                  TEXT       PRIMARY KEY,
  "client_id"             TEXT       NOT NULL,
  "user_id"               TEXT       NOT NULL,
  "redirect_uri"          TEXT       NOT NULL,
  "code_challenge"        TEXT       NOT NULL,
  "code_challenge_method" TEXT       NOT NULL DEFAULT 'S256',
  "scope"                 TEXT[]     NOT NULL DEFAULT '{}'::text[],
  "expires_at"            TIMESTAMP  NOT NULL,
  "consumed_at"           TIMESTAMP,
  "created_at"            TIMESTAMP  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS "LiteLLM_OAuthAuthorizationCode_client_expires_idx"
  ON "LiteLLM_OAuthAuthorizationCode" ("client_id", "expires_at");


ALTER TABLE "LiteLLM_VerificationToken"
  ADD COLUMN IF NOT EXISTS "app_id"      TEXT,
  ADD COLUMN IF NOT EXISTS "token_type"  TEXT;
-- token_type values:
--   NULL           legacy admin-provisioned virtual key (existing rows)
--   "oauth_access" issued via /oauth/token grant_type=authorization_code
--   "oauth_refresh" issued via /oauth/token (matching refresh token row)

CREATE INDEX IF NOT EXISTS "LiteLLM_VerificationToken_app_id_idx"
  ON "LiteLLM_VerificationToken" ("app_id")
  WHERE "app_id" IS NOT NULL;
