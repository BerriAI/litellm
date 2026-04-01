-- Add total_spend column to LiteLLM_TeamMembership
-- Tracks lifetime (never-zeroed) spend for a user within a team,
-- independent of the current-period spend that resets periodically.
ALTER TABLE "LiteLLM_TeamMembership" ADD COLUMN IF NOT EXISTS "total_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0;
