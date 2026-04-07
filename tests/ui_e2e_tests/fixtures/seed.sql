-- UI E2E Test Database Seed
-- Run with: psql $DATABASE_URL -f seed.sql

-- ============================================================
-- 1. Budget Table (must be first — referenced by org FK)
-- ============================================================
INSERT INTO "LiteLLM_BudgetTable" (
    budget_id, max_budget, created_by, updated_by
) VALUES (
    'e2e-budget-org', 1000.0, 'e2e-proxy-admin', 'e2e-proxy-admin'
) ON CONFLICT (budget_id) DO NOTHING;

-- ============================================================
-- 2. Organization
-- ============================================================
INSERT INTO "LiteLLM_OrganizationTable" (
    organization_id, organization_alias, budget_id, metadata, models, spend,
    model_spend, created_by, updated_by
) VALUES (
    'e2e-org-main', 'E2E Organization', 'e2e-budget-org', '{}'::jsonb,
    ARRAY[]::text[], 0.0, '{}'::jsonb, 'e2e-proxy-admin', 'e2e-proxy-admin'
) ON CONFLICT (organization_id) DO NOTHING;

-- ============================================================
-- 3. Users (password is scrypt hash of "test")
-- ============================================================
INSERT INTO "LiteLLM_UserTable" (
    user_id, user_email, user_role, password, teams, models, metadata,
    spend, model_spend, model_max_budget
) VALUES
(
    'e2e-proxy-admin', 'admin@test.local', 'proxy_admin', 'scrypt:MU5CcTAi6rVK1HfY1rVPEWq6r4sxg837eq9dG4n5Q6BhDJ44442+seC6LAhLEAYr',
    ARRAY['e2e-team-crud']::text[], ARRAY[]::text[], '{}'::jsonb,
    0.0, '{}'::jsonb, '{}'::jsonb
),
(
    'e2e-admin-viewer', 'adminviewer@test.local', 'proxy_admin_viewer', 'scrypt:MU5CcTAi6rVK1HfY1rVPEWq6r4sxg837eq9dG4n5Q6BhDJ44442+seC6LAhLEAYr',
    ARRAY[]::text[], ARRAY[]::text[], '{}'::jsonb,
    0.0, '{}'::jsonb, '{}'::jsonb
),
(
    'e2e-internal-user', 'internal@test.local', 'internal_user', 'scrypt:MU5CcTAi6rVK1HfY1rVPEWq6r4sxg837eq9dG4n5Q6BhDJ44442+seC6LAhLEAYr',
    ARRAY['e2e-team-crud', 'e2e-team-org']::text[], ARRAY[]::text[], '{}'::jsonb,
    0.0, '{}'::jsonb, '{}'::jsonb
),
(
    'e2e-internal-viewer', 'viewer@test.local', 'internal_user_viewer', 'scrypt:MU5CcTAi6rVK1HfY1rVPEWq6r4sxg837eq9dG4n5Q6BhDJ44442+seC6LAhLEAYr',
    ARRAY[]::text[], ARRAY[]::text[], '{}'::jsonb,
    0.0, '{}'::jsonb, '{}'::jsonb
),
(
    'e2e-team-admin', 'teamadmin@test.local', 'internal_user', 'scrypt:MU5CcTAi6rVK1HfY1rVPEWq6r4sxg837eq9dG4n5Q6BhDJ44442+seC6LAhLEAYr',
    ARRAY['e2e-team-crud', 'e2e-team-delete']::text[], ARRAY[]::text[], '{}'::jsonb,
    0.0, '{}'::jsonb, '{}'::jsonb
)
ON CONFLICT (user_id) DO NOTHING;

-- ============================================================
-- 4. Teams
-- ============================================================
INSERT INTO "LiteLLM_TeamTable" (
    team_id, team_alias, organization_id, admins, members,
    members_with_roles, metadata, models, spend, model_spend,
    model_max_budget, blocked
) VALUES
(
    'e2e-team-crud', 'E2E Team CRUD', NULL,
    ARRAY['e2e-team-admin']::text[],
    ARRAY['e2e-team-admin', 'e2e-internal-user']::text[],
    '[{"role": "admin", "user_id": "e2e-team-admin"}, {"role": "user", "user_id": "e2e-internal-user"}]'::jsonb,
    '{}'::jsonb,
    ARRAY['fake-openai-gpt-4', 'fake-anthropic-claude']::text[],
    0.0, '{}'::jsonb, '{}'::jsonb, false
),
(
    'e2e-team-delete', 'E2E Team Delete', NULL,
    ARRAY['e2e-team-admin']::text[],
    ARRAY['e2e-team-admin']::text[],
    '[{"role": "admin", "user_id": "e2e-team-admin"}]'::jsonb,
    '{}'::jsonb,
    ARRAY['fake-openai-gpt-4']::text[],
    0.0, '{}'::jsonb, '{}'::jsonb, false
),
(
    'e2e-team-org', 'E2E Team In Org', 'e2e-org-main',
    ARRAY[]::text[],
    ARRAY['e2e-internal-user']::text[],
    '[{"role": "user", "user_id": "e2e-internal-user"}]'::jsonb,
    '{}'::jsonb,
    ARRAY['fake-openai-gpt-4']::text[],
    0.0, '{}'::jsonb, '{}'::jsonb, false
)
ON CONFLICT (team_id) DO NOTHING;

-- ============================================================
-- 5. Team Memberships
-- ============================================================
INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id, spend) VALUES
    ('e2e-team-admin', 'e2e-team-crud', 0.0),
    ('e2e-internal-user', 'e2e-team-crud', 0.0),
    ('e2e-team-admin', 'e2e-team-delete', 0.0),
    ('e2e-internal-user', 'e2e-team-org', 0.0)
ON CONFLICT (user_id, team_id) DO NOTHING;
