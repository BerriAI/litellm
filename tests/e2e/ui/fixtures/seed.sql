-- E2E Test Seed Data
-- Idempotent: deletes all e2e-* rows then re-inserts deterministic data.

-- 1. Clean up in dependency order
DELETE FROM "LiteLLM_SpendLogs" WHERE request_id = 'e2e-app-forms-request';
DELETE FROM "LiteLLM_DailyCustomerMemberSpend" WHERE id = 'e2e-app-forms-spend';
DELETE FROM "LiteLLM_AppCustomerMemberTable" WHERE app_id = 'e2e-app-forms';
DELETE FROM "LiteLLM_AppCustomerOrgTable" WHERE app_id = 'e2e-app-forms';
DELETE FROM "LiteLLM_AppTable" WHERE app_id = 'e2e-app-forms';
DELETE FROM "LiteLLM_AppGroupTable" WHERE app_group_id = 'e2e-app-forms-group';
DELETE FROM "LiteLLM_CustomerMemberTable" WHERE customer_member_id = 'e2e-app-forms-member';
DELETE FROM "LiteLLM_CustomerTeamTable" WHERE customer_team_id = 'e2e-app-forms-customer-team';
DELETE FROM "LiteLLM_CustomerOrganizationTable" WHERE customer_org_id = 'e2e-app-forms-customer';
DELETE FROM "LiteLLM_InvitationLink"
WHERE "user_id" LIKE 'e2e-%' OR "created_by" LIKE 'e2e-%' OR "updated_by" LIKE 'e2e-%';
DELETE FROM "LiteLLM_TeamMembership" WHERE "user_id" LIKE 'e2e-%';
DELETE FROM "LiteLLM_VerificationToken" WHERE token LIKE 'e2e-%';
DELETE FROM "LiteLLM_TeamTable" WHERE "team_id" LIKE 'e2e-%';
DELETE FROM "LiteLLM_OrganizationTable" WHERE "organization_id" LIKE 'e2e-%';
DELETE FROM "LiteLLM_UserTable" WHERE "user_id" LIKE 'e2e-%';
DELETE FROM "LiteLLM_BudgetTable" WHERE "budget_id" LIKE 'e2e-%';
DELETE FROM "LiteLLM_ObjectPermissionTable" WHERE object_permission_id IN
  ('e2e-app-forms-team-grant', 'e2e-app-forms-user-grant', 'e2e-app-forms-org-grant');

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

-- 4. Users (password hash is scrypt of E2E_SEEDED_USER_PASSWORD from constants.ts)
INSERT INTO "LiteLLM_UserTable" ("user_id", "user_email", "user_role", "teams", "password")
VALUES
  ('e2e-proxy-admin',      'admin@test.local',       'proxy_admin',           '{"e2e-team-crud"}',                  'scrypt:KdnTJwPb3gswdqSznPE5CC6apeFIMycd6BG7yRWndZa3QZPcVs37y7jvrQCaPUNq'),
  ('e2e-admin-viewer',     'adminviewer@test.local',  'proxy_admin_viewer',   '{}',                                 'scrypt:KdnTJwPb3gswdqSznPE5CC6apeFIMycd6BG7yRWndZa3QZPcVs37y7jvrQCaPUNq'),
  ('e2e-internal-user',    'internal@test.local',     'internal_user',        '{"e2e-team-crud","e2e-team-org","e2e-team-keygen"}', 'scrypt:KdnTJwPb3gswdqSznPE5CC6apeFIMycd6BG7yRWndZa3QZPcVs37y7jvrQCaPUNq'),
  ('e2e-internal-viewer',  'viewer@test.local',       'internal_user_viewer', '{"e2e-team-crud"}',                  'scrypt:KdnTJwPb3gswdqSznPE5CC6apeFIMycd6BG7yRWndZa3QZPcVs37y7jvrQCaPUNq'),
  ('e2e-team-admin',       'teamadmin@test.local',    'internal_user',        '{"e2e-team-crud","e2e-team-delete"}', 'scrypt:KdnTJwPb3gswdqSznPE5CC6apeFIMycd6BG7yRWndZa3QZPcVs37y7jvrQCaPUNq'),
  ('e2e-invitable-user',   'invitable@test.local',    'internal_user',        '{}',                                 'scrypt:KdnTJwPb3gswdqSznPE5CC6apeFIMycd6BG7yRWndZa3QZPcVs37y7jvrQCaPUNq'),
  ('e2e-internal-noteam',  'noteam@test.local',       'internal_user',        '{}',                                 'scrypt:KdnTJwPb3gswdqSznPE5CC6apeFIMycd6BG7yRWndZa3QZPcVs37y7jvrQCaPUNq'),
  ('e2e-invitable-by-team-admin', 'invitable-team@test.local', 'internal_user', '{}',                                'scrypt:KdnTJwPb3gswdqSznPE5CC6apeFIMycd6BG7yRWndZa3QZPcVs37y7jvrQCaPUNq'),
  ('e2e-removable-member', 'removable@test.local',    'internal_user',        '{"e2e-team-crud"}',                  'scrypt:KdnTJwPb3gswdqSznPE5CC6apeFIMycd6BG7yRWndZa3QZPcVs37y7jvrQCaPUNq');

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

INSERT INTO "LiteLLM_TeamTable" (
  "team_id", "team_alias", "organization_id", "admins", "members",
  "members_with_roles", "metadata", "models", "spend", "model_spend", "model_max_budget", "blocked",
  "team_member_permissions"
) VALUES
  ('e2e-team-keygen', 'E2E Team Keygen', NULL,
   '{}', '{"e2e-internal-user"}',
   '[{"role":"user","user_id":"e2e-internal-user"}]'::jsonb,
   '{}'::jsonb, '{"fake-openai-gpt-4"}', 0.0, '{}'::jsonb, '{}'::jsonb, false,
   '{"/key/generate"}');

-- 6. Team Memberships (only user_id, team_id, spend — no created_at/updated_at)
INSERT INTO "LiteLLM_TeamMembership" ("user_id", "team_id", "spend")
VALUES
  ('e2e-team-admin',       'e2e-team-crud',     0.0),
  ('e2e-internal-user',    'e2e-team-crud',     0.0),
  ('e2e-internal-viewer',  'e2e-team-crud',     0.0),
  ('e2e-removable-member', 'e2e-team-crud',     0.0),
  ('e2e-team-admin',       'e2e-team-delete',   0.0),
  ('e2e-internal-user',    'e2e-team-org',      0.0),
  ('e2e-internal-user',    'e2e-team-keygen',   0.0),
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

INSERT INTO "LiteLLM_Config" (param_name, param_value)
VALUES ('customer_app_definition', '{"resource_type":"team","auto_create_customers_without_app":true}')
ON CONFLICT (param_name) DO NOTHING;

INSERT INTO "LiteLLM_ObjectPermissionTable" (object_permission_id, apps) VALUES
  ('e2e-app-forms-team-grant', ARRAY['e2e-app-forms']),
  ('e2e-app-forms-user-grant', ARRAY['e2e-app-forms']),
  ('e2e-app-forms-org-grant', ARRAY['e2e-app-forms']);
INSERT INTO "LiteLLM_TeamTable" (team_id, team_alias, admins, members, members_with_roles, models, object_permission_id) VALUES
  ('e2e-app-forms-team', 'E2E Forms App', '{}', '{}', '[]', '{"fake-openai-gpt-4"}', 'e2e-app-forms-team-grant');
INSERT INTO "LiteLLM_UserTable" (user_id, user_alias, user_email, user_role, object_permission_id) VALUES
  ('e2e-app-forms-user', 'E2E Forms Grantee', 'app-forms-grantee@test.local', 'internal_user', 'e2e-app-forms-user-grant');
INSERT INTO "LiteLLM_OrganizationTable" (organization_id, organization_alias, budget_id, object_permission_id, created_by, updated_by) VALUES
  ('e2e-app-forms-org', 'E2E Forms Organization', 'e2e-budget-org', 'e2e-app-forms-org-grant', 'e2e-proxy-admin', 'e2e-proxy-admin');

INSERT INTO "LiteLLM_AppGroupTable" (app_group_id, name, created_by, updated_by) VALUES
  ('e2e-app-forms-group', 'E2E Forms App Group', 'e2e-proxy-admin', 'e2e-proxy-admin');
INSERT INTO "LiteLLM_AppTable" (app_id, alias, resource_type, resource_id, app_group_id, customer_margin, created_by, updated_by) VALUES
  ('e2e-app-forms', 'E2E Forms App', 'team', 'e2e-app-forms-team', 'e2e-app-forms-group', 0.10, 'e2e-proxy-admin', 'e2e-proxy-admin');
INSERT INTO "LiteLLM_CustomerOrganizationTable" (customer_org_id, alias, created_by, updated_by) VALUES
  ('e2e-app-forms-customer', 'E2E Forms Customer', 'e2e-proxy-admin', 'e2e-proxy-admin');
INSERT INTO "LiteLLM_CustomerTeamTable" (customer_team_id, alias, parent_org_id, created_by, updated_by) VALUES
  ('e2e-app-forms-customer-team', 'E2E Forms Customer Team', 'e2e-app-forms-customer', 'e2e-proxy-admin', 'e2e-proxy-admin');
INSERT INTO "LiteLLM_CustomerMemberTable" (customer_member_id, alias, parent_org_id, parent_team_ids, created_by, updated_by) VALUES
  ('e2e-app-forms-member', 'E2E Forms Member', 'e2e-app-forms-customer', ARRAY['e2e-app-forms-customer-team'], 'e2e-proxy-admin', 'e2e-proxy-admin');
INSERT INTO "LiteLLM_BudgetTable" (budget_id, max_budget, created_by, updated_by) VALUES
  ('e2e-app-forms-org-budget', 50, 'e2e-proxy-admin', 'e2e-proxy-admin'),
  ('e2e-app-forms-member-budget', 50, 'e2e-proxy-admin', 'e2e-proxy-admin');
INSERT INTO "LiteLLM_AppCustomerOrgTable" (app_id, customer_org_id, budget_id) VALUES
  ('e2e-app-forms', 'e2e-app-forms-customer', 'e2e-app-forms-org-budget');
INSERT INTO "LiteLLM_AppCustomerMemberTable" (app_id, customer_member_id, budget_id) VALUES
  ('e2e-app-forms', 'e2e-app-forms-member', 'e2e-app-forms-member-budget');
INSERT INTO "LiteLLM_DailyCustomerMemberSpend" (id, date, app_id, app_group_id, customer_org_id, customer_team_id, customer_member_id, spend, api_requests, successful_requests, updated_at) VALUES
  ('e2e-app-forms-spend', '2026-01-15', 'e2e-app-forms', 'e2e-app-forms-group', 'e2e-app-forms-customer', 'e2e-app-forms-customer-team', 'e2e-app-forms-member', 3, 1, 1, CURRENT_TIMESTAMP);
INSERT INTO "LiteLLM_SpendLogs" (request_id, call_type, spend, "startTime", "endTime", end_user, team_id, metadata) VALUES
  ('e2e-app-forms-request', 'completion', 3, '2026-01-15T10:00:00Z', '2026-01-15T10:00:01Z',
   'e2e-app-forms-member', 'e2e-app-forms-team',
   '{"app_id":"e2e-app-forms","app_group_id":"e2e-app-forms-group","customer_org_id":"e2e-app-forms-customer","customer_member_id":"e2e-app-forms-member"}');
