-- =============================================================================
-- LiteLLM Test Data Seed — last 7 days
-- Simulates 3 teams, 6 users, 3 API keys, 3 models
-- Run after LiteLLM has applied its migrations:
--   docker compose -f docker-compose.test.yml exec db \
--     psql -U llmproxy -d litellm -f /seed_test_data.sql
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Teams
-- -----------------------------------------------------------------------------
INSERT INTO "LiteLLM_TeamTable" (team_id, team_alias, spend, models, members_with_roles, metadata, blocked, created_at, updated_at)
VALUES
  ('team-platform-001', 'Platform Engineering', 0.0, ARRAY['gpt-4o','gpt-4o-mini','claude-3-5-sonnet'], '{}', '{}', false, NOW(), NOW()),
  ('team-data-001',     'Data Science',         0.0, ARRAY['gpt-4o','gpt-4o-mini'],                    '{}', '{}', false, NOW(), NOW()),
  ('team-product-001',  'Product',              0.0, ARRAY['gpt-4o-mini','claude-3-5-sonnet'],          '{}', '{}', false, NOW(), NOW())
ON CONFLICT (team_id) DO NOTHING;

-- -----------------------------------------------------------------------------
-- Users
-- -----------------------------------------------------------------------------
INSERT INTO "LiteLLM_UserTable" (user_id, user_email, team_id, spend, models, teams, metadata)
VALUES
  ('user-alice',   'alice@example.com',   'team-platform-001', 0.0, ARRAY[]::text[], ARRAY['team-platform-001'], '{}'),
  ('user-bob',     'bob@example.com',     'team-platform-001', 0.0, ARRAY[]::text[], ARRAY['team-platform-001'], '{}'),
  ('user-carol',   'carol@example.com',   'team-data-001',     0.0, ARRAY[]::text[], ARRAY['team-data-001'],     '{}'),
  ('user-dave',    'dave@example.com',    'team-data-001',     0.0, ARRAY[]::text[], ARRAY['team-data-001'],     '{}'),
  ('user-eve',     'eve@example.com',     'team-product-001',  0.0, ARRAY[]::text[], ARRAY['team-product-001'],  '{}'),
  ('user-frank',   'frank@example.com',   'team-product-001',  0.0, ARRAY[]::text[], ARRAY['team-product-001'],  '{}')
ON CONFLICT (user_id) DO NOTHING;

-- -----------------------------------------------------------------------------
-- API Keys (VerificationToken)
-- token value is what goes in the api_key column of DailyUserSpend
-- -----------------------------------------------------------------------------
INSERT INTO "LiteLLM_VerificationToken" (token, key_alias, key_name, user_id, team_id, spend, models, aliases, config, permissions, metadata)
VALUES
  ('sk-platform-key-hashed', 'platform-key',  'Platform Key',  'user-alice', 'team-platform-001', 0.0, ARRAY['gpt-4o','gpt-4o-mini','claude-3-5-sonnet'], '{}', '{}', '{}', '{}'),
  ('sk-data-key-hashed',     'data-key',      'Data Key',      'user-carol', 'team-data-001',     0.0, ARRAY['gpt-4o','gpt-4o-mini'],                    '{}', '{}', '{}', '{}'),
  ('sk-product-key-hashed',  'product-key',   'Product Key',   'user-eve',   'team-product-001',  0.0, ARRAY['gpt-4o-mini','claude-3-5-sonnet'],          '{}', '{}', '{}', '{}')
ON CONFLICT (token) DO NOTHING;

-- -----------------------------------------------------------------------------
-- Daily spend data — last 7 days
-- One row per (user, api_key, model, date)
-- -----------------------------------------------------------------------------

-- Helper: generate dates for last 7 days
WITH dates AS (
  SELECT generate_series(
    (CURRENT_DATE - INTERVAL '6 days')::date,
    (CURRENT_DATE - INTERVAL '1 day')::date,
    '1 day'::interval
  )::date AS day
),

spend_rows AS (
  -- Platform team: Alice using gpt-4o (heavy usage)
  SELECT d.day::text AS date, 'user-alice' AS user_id, 'sk-platform-key-hashed' AS api_key,
    'gpt-4o' AS model, 'gpt-4o' AS model_group, 'openai' AS custom_llm_provider,
    (1200 + (random()*400)::int)::bigint  AS prompt_tokens,
    (400  + (random()*200)::int)::bigint  AS completion_tokens,
    0::bigint AS cache_read_input_tokens, 0::bigint AS cache_creation_input_tokens,
    round((0.015 + random()*0.01)::numeric, 6)::float AS spend,
    (8 + (random()*4)::int)::bigint AS api_requests,
    (8 + (random()*4)::int)::bigint AS successful_requests,
    0::bigint AS failed_requests
  FROM dates d

  UNION ALL

  -- Platform team: Bob using gpt-4o-mini
  SELECT d.day::text, 'user-bob', 'sk-platform-key-hashed',
    'gpt-4o-mini', 'gpt-4o-mini', 'openai',
    (800  + (random()*300)::int)::bigint,
    (200  + (random()*100)::int)::bigint,
    0::bigint, 0::bigint,
    round((0.002 + random()*0.002)::numeric, 6)::float,
    (15 + (random()*10)::int)::bigint,
    (15 + (random()*10)::int)::bigint,
    0::bigint
  FROM dates d

  UNION ALL

  -- Platform team: Alice using claude-3-5-sonnet (with cache tokens)
  SELECT d.day::text, 'user-alice', 'sk-platform-key-hashed',
    'claude-3-5-sonnet', 'claude-3-5-sonnet', 'anthropic',
    (600  + (random()*200)::int)::bigint,
    (300  + (random()*100)::int)::bigint,
    (100  + (random()*50)::int)::bigint,
    (50   + (random()*20)::int)::bigint,
    round((0.012 + random()*0.008)::numeric, 6)::float,
    (5 + (random()*3)::int)::bigint,
    (5 + (random()*3)::int)::bigint,
    0::bigint
  FROM dates d

  UNION ALL

  -- Data team: Carol using gpt-4o (batch analysis)
  SELECT d.day::text, 'user-carol', 'sk-data-key-hashed',
    'gpt-4o', 'gpt-4o', 'openai',
    (2000 + (random()*800)::int)::bigint,
    (600  + (random()*300)::int)::bigint,
    0::bigint, 0::bigint,
    round((0.025 + random()*0.015)::numeric, 6)::float,
    (12 + (random()*6)::int)::bigint,
    (11 + (random()*5)::int)::bigint,
    (0 + (random()*2)::int)::bigint
  FROM dates d

  UNION ALL

  -- Data team: Dave using gpt-4o-mini
  SELECT d.day::text, 'user-dave', 'sk-data-key-hashed',
    'gpt-4o-mini', 'gpt-4o-mini', 'openai',
    (1500 + (random()*500)::int)::bigint,
    (400  + (random()*150)::int)::bigint,
    0::bigint, 0::bigint,
    round((0.004 + random()*0.003)::numeric, 6)::float,
    (20 + (random()*10)::int)::bigint,
    (20 + (random()*10)::int)::bigint,
    0::bigint
  FROM dates d

  UNION ALL

  -- Product team: Eve using claude-3-5-sonnet
  SELECT d.day::text, 'user-eve', 'sk-product-key-hashed',
    'claude-3-5-sonnet', 'claude-3-5-sonnet', 'anthropic',
    (900  + (random()*300)::int)::bigint,
    (350  + (random()*150)::int)::bigint,
    (80   + (random()*40)::int)::bigint,
    0::bigint,
    round((0.018 + random()*0.010)::numeric, 6)::float,
    (6 + (random()*4)::int)::bigint,
    (6 + (random()*4)::int)::bigint,
    0::bigint
  FROM dates d

  UNION ALL

  -- Product team: Frank using gpt-4o-mini
  SELECT d.day::text, 'user-frank', 'sk-product-key-hashed',
    'gpt-4o-mini', 'gpt-4o-mini', 'openai',
    (500  + (random()*200)::int)::bigint,
    (150  + (random()*80)::int)::bigint,
    0::bigint, 0::bigint,
    round((0.001 + random()*0.001)::numeric, 6)::float,
    (10 + (random()*5)::int)::bigint,
    (10 + (random()*5)::int)::bigint,
    0::bigint
  FROM dates d
)

INSERT INTO "LiteLLM_DailyUserSpend" (
  id, date, user_id, api_key, model, model_group, custom_llm_provider,
  prompt_tokens, completion_tokens, cache_read_input_tokens, cache_creation_input_tokens,
  spend, api_requests, successful_requests, failed_requests,
  created_at, updated_at
)
SELECT
  gen_random_uuid()::text,
  date, user_id, api_key, model, model_group, custom_llm_provider,
  prompt_tokens, completion_tokens, cache_read_input_tokens, cache_creation_input_tokens,
  spend, api_requests, successful_requests, failed_requests,
  NOW(), NOW()
FROM spend_rows
ON CONFLICT (user_id, date, api_key, model, custom_llm_provider, mcp_namespaced_tool_name, endpoint) DO NOTHING;

-- -----------------------------------------------------------------------------
-- Roll up spend totals to parent tables so UI budget bars are accurate
-- -----------------------------------------------------------------------------
UPDATE "LiteLLM_VerificationToken" t
SET spend = (
  SELECT COALESCE(SUM(d.spend), 0)
  FROM "LiteLLM_DailyUserSpend" d
  WHERE d.api_key = t.token
)
WHERE t.token IN ('sk-platform-key-hashed','sk-data-key-hashed','sk-product-key-hashed');

UPDATE "LiteLLM_UserTable" u
SET spend = (
  SELECT COALESCE(SUM(d.spend), 0)
  FROM "LiteLLM_DailyUserSpend" d
  WHERE d.user_id = u.user_id
);

UPDATE "LiteLLM_TeamTable" tt
SET spend = (
  SELECT COALESCE(SUM(d.spend), 0)
  FROM "LiteLLM_DailyUserSpend" d
  JOIN "LiteLLM_VerificationToken" vt ON d.api_key = vt.token
  WHERE vt.team_id = tt.team_id
);

-- Confirm
SELECT
  date,
  COUNT(*)           AS rows,
  SUM(spend)::numeric(10,4) AS total_spend,
  SUM(api_requests)  AS total_requests
FROM "LiteLLM_DailyUserSpend"
GROUP BY date
ORDER BY date;
