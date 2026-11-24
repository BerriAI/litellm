ALTER TABLE "LiteLLM_AutoRouterSession"
ADD COLUMN IF NOT EXISTS "savings_estimated_turns" INTEGER NOT NULL DEFAULT 0,
ADD COLUMN IF NOT EXISTS "savings_estimated_actual_spend" DOUBLE PRECISION NOT NULL DEFAULT 0,
ADD COLUMN IF NOT EXISTS "savings_estimated_saved_spend" DOUBLE PRECISION NOT NULL DEFAULT 0,
ADD COLUMN IF NOT EXISTS "savings_estimated_baseline_models" JSONB NOT NULL DEFAULT '{}';
