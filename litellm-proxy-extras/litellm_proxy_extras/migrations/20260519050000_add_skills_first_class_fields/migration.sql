-- Migration: add_skills_first_class_fields  (S2-02)
--
-- Adds nullable columns to LiteLLM_SkillsTable so xct-native skills become a
-- first-class entity: scoped per team/user, versioned, permission-gated,
-- with a structured tool_schema + system_prompt_template.
--
-- Strictly additive — existing Anthropic (source='anthropic') rows are not
-- touched. All new columns are nullable or have safe defaults.

ALTER TABLE "LiteLLM_SkillsTable"
  ADD COLUMN IF NOT EXISTS "team_id"                 TEXT,
  ADD COLUMN IF NOT EXISTS "user_id"                 TEXT,
  ADD COLUMN IF NOT EXISTS "version"                 TEXT,
  ADD COLUMN IF NOT EXISTS "is_public"               BOOLEAN NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS "object_permission_id"    TEXT,
  ADD COLUMN IF NOT EXISTS "tool_schema"             JSONB,
  ADD COLUMN IF NOT EXISTS "system_prompt_template"  TEXT,
  ADD COLUMN IF NOT EXISTS "xct_metadata"            JSONB NOT NULL DEFAULT '{}'::jsonb;

-- Capability-discovery hot path (xct rows scoped to a team).
CREATE INDEX IF NOT EXISTS "LiteLLM_SkillsTable_source_team_id_idx"
  ON "LiteLLM_SkillsTable" ("source", "team_id");

-- Owner lookup (xct rows scoped to a user).
CREATE INDEX IF NOT EXISTS "LiteLLM_SkillsTable_source_user_id_idx"
  ON "LiteLLM_SkillsTable" ("source", "user_id");

-- Public skills lookup (anonymous /well-known/xct-capabilities).
CREATE INDEX IF NOT EXISTS "LiteLLM_SkillsTable_source_is_public_idx"
  ON "LiteLLM_SkillsTable" ("source", "is_public")
  WHERE "is_public" IS TRUE;
