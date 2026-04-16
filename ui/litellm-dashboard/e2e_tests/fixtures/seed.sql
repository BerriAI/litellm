-- E2E Test Seed Data
-- Idempotent: deletes all e2e-* rows then re-inserts deterministic data.

-- 1. Clean up in dependency order
DELETE FROM "LiteLLM_TeamMembership" WHERE "user_id" LIKE 'e2e-%';
DELETE FROM "LiteLLM_VerificationToken" WHERE token LIKE 'e2e-%';
DELETE FROM "LiteLLM_TeamTable" WHERE "team_id" LIKE 'e2e-%';
DELETE FROM "LiteLLM_OrganizationTable" WHERE "organization_id" LIKE 'e2e-%';
DELETE FROM "LiteLLM_UserTable" WHERE "user_id" LIKE 'e2e-%';
DELETE FROM "LiteLLM_BudgetTable" WHERE "budget_id" LIKE 'e2e-%';

-- 2. Budget (created_by and updated_by are NOT NULL)
INSERT INTO "LiteLLM_BudgetTable" ("budget_id", "max_budget", "created_by", "updated_by")
VALUES ('e2e-budget-org', 1000, 'e2e-proxy-admin', 'e2e-proxy-admin');

-- 3. Organization (created_by and updated_by are NOT NULL)
INSERT INTO "LiteLLM_OrganizationTable" (
  "organization_id", "organization_alias", "budget_id",
  "metadata", "models", "spend", "model_spend",
  "created_by", "updated_by"
) VALUES (
  'e2e-org-main', 'E2E Organization', 'e2e-budget-org',
  '{}'::jsonb, ARRAY[]::text[], 0.0, '{}'::jsonb,
  'e2e-proxy-admin', 'e2e-proxy-admin'
);

-- 4. Users (password hash is scrypt of "test")
INSERT INTO "LiteLLM_UserTable" ("user_id", "user_email", "user_role", "teams", "password")
VALUES
  ('e2e-proxy-admin',      'admin@test.local',       'proxy_admin',           '{"e2e-team-crud"}',                  'scrypt:MU5CcTAi6rVK1HfY1rVPEWq6r4sxg837eq9dG4n5Q6BhDJ44442+seC6LAhLEAYr'),
  ('e2e-admin-viewer',     'adminviewer@test.local',  'proxy_admin_viewer',   '{}',                                 'scrypt:MU5CcTAi6rVK1HfY1rVPEWq6r4sxg837eq9dG4n5Q6BhDJ44442+seC6LAhLEAYr'),
  ('e2e-internal-user',    'internal@test.local',     'internal_user',        '{"e2e-team-crud","e2e-team-org"}',   'scrypt:MU5CcTAi6rVK1HfY1rVPEWq6r4sxg837eq9dG4n5Q6BhDJ44442+seC6LAhLEAYr'),
  ('e2e-internal-viewer',  'viewer@test.local',       'internal_user_viewer', '{"e2e-team-crud"}',                  'scrypt:MU5CcTAi6rVK1HfY1rVPEWq6r4sxg837eq9dG4n5Q6BhDJ44442+seC6LAhLEAYr'),
  ('e2e-team-admin',       'teamadmin@test.local',    'internal_user',        '{"e2e-team-crud","e2e-team-delete"}', 'scrypt:MU5CcTAi6rVK1HfY1rVPEWq6r4sxg837eq9dG4n5Q6BhDJ44442+seC6LAhLEAYr'),
  ('e2e-invitable-user',   'invitable@test.local',    'internal_user',        '{}',                                 'scrypt:MU5CcTAi6rVK1HfY1rVPEWq6r4sxg837eq9dG4n5Q6BhDJ44442+seC6LAhLEAYr'),
  ('e2e-removable-member', 'removable@test.local',    'internal_user',        '{"e2e-team-crud"}',                  'scrypt:MU5CcTAi6rVK1HfY1rVPEWq6r4sxg837eq9dG4n5Q6BhDJ44442+seC6LAhLEAYr');

-- 5. Teams (members_with_roles is required JSON)
INSERT INTO "LiteLLM_TeamTable" (
  "team_id", "team_alias", "organization_id", "admins", "members",
  "members_with_roles", "metadata", "models", "spend", "model_spend", "model_max_budget", "blocked"
) VALUES
  ('e2e-team-crud', 'E2E Team CRUD', NULL,
   '{"e2e-team-admin"}',
   '{"e2e-team-admin","e2e-internal-user","e2e-internal-viewer","e2e-removable-member"}',
   '[{"role":"admin","user_id":"e2e-team-admin"},{"role":"user","user_id":"e2e-internal-user"},{"role":"user","user_id":"e2e-internal-viewer"},{"role":"user","user_id":"e2e-removable-member"}]'::jsonb,
   '{}'::jsonb, '{"fake-openai-gpt-4","fake-anthropic-claude"}', 0.0, '{}'::jsonb, '{}'::jsonb, false),

  ('e2e-team-delete', 'E2E Team Delete', NULL,
   '{"e2e-team-admin"}', '{"e2e-team-admin"}',
   '[{"role":"admin","user_id":"e2e-team-admin"}]'::jsonb,
   '{}'::jsonb, '{"fake-openai-gpt-4"}', 0.0, '{}'::jsonb, '{}'::jsonb, false),

  ('e2e-team-org', 'E2E Team In Org', 'e2e-org-main',
   '{}', '{"e2e-internal-user"}',
   '[{"role":"user","user_id":"e2e-internal-user"}]'::jsonb,
   '{}'::jsonb, '{"fake-openai-gpt-4"}', 0.0, '{}'::jsonb, '{}'::jsonb, false),

  ('e2e-team-no-admin', 'E2E Team No Admin', NULL,
   '{}', '{"e2e-invitable-user"}',
   '[{"role":"user","user_id":"e2e-invitable-user"}]'::jsonb,
   '{}'::jsonb, '{"fake-openai-gpt-4"}', 0.0, '{}'::jsonb, '{}'::jsonb, false);

-- 6. Team Memberships (only user_id, team_id, spend — no created_at/updated_at)
INSERT INTO "LiteLLM_TeamMembership" ("user_id", "team_id", "spend")
VALUES
  ('e2e-team-admin',       'e2e-team-crud',     0.0),
  ('e2e-internal-user',    'e2e-team-crud',     0.0),
  ('e2e-internal-viewer',  'e2e-team-crud',     0.0),
  ('e2e-removable-member', 'e2e-team-crud',     0.0),
  ('e2e-team-admin',       'e2e-team-delete',   0.0),
  ('e2e-internal-user',    'e2e-team-org',      0.0),
  ('e2e-invitable-user',   'e2e-team-no-admin', 0.0);

-- 7. Verification Tokens (API Keys)
INSERT INTO "LiteLLM_VerificationToken" (
  "token", "key_name", "key_alias", "user_id", "team_id",
  "models", "spend", "max_budget", "expires", "metadata"
) VALUES
  ('e2e-key-update-limits',  'sk-e2e-update',   'e2eUpdateLimitsKey',  'e2e-proxy-admin',    'e2e-team-crud', '{"fake-openai-gpt-4"}', 0.0, NULL, NULL, '{}'::jsonb),
  ('e2e-key-delete',         'sk-e2e-delete',    'e2eDeleteKey',        'e2e-proxy-admin',    'e2e-team-crud', '{"fake-openai-gpt-4"}', 0.0, NULL, NULL, '{}'::jsonb),
  ('e2e-key-regenerate',     'sk-e2e-regen',     'e2eRegenerateKey',    'e2e-proxy-admin',    'e2e-team-crud', '{"fake-openai-gpt-4"}', 0.0, NULL, NULL, '{}'::jsonb),
  ('e2e-key-internal-user',  'sk-e2e-internal',  'e2eInternalUserKey',  'e2e-internal-user',  'e2e-team-crud', '{"fake-openai-gpt-4"}', 0.0, NULL, NULL, '{}'::jsonb),
  ('e2e-key-viewer',         'sk-e2e-viewer',    'e2eViewerKey',        'e2e-internal-viewer', NULL,           '{"fake-openai-gpt-4"}', 0.0, NULL, NULL, '{}'::jsonb);
