SET LOCAL lock_timeout = '10s';

-- AlterTable
-- Cumulative per-team token budget, mirroring the existing dollar budget
-- (max_budget / spend). max_tokens_budget is the cap (null = untracked);
-- tokens_used is the accumulator reset on the shared budget_reset_at tick.
ALTER TABLE "LiteLLM_TeamTable" ADD COLUMN IF NOT EXISTS "max_tokens_budget" BIGINT;
ALTER TABLE "LiteLLM_TeamTable" ADD COLUMN IF NOT EXISTS "tokens_used" BIGINT NOT NULL DEFAULT 0;
