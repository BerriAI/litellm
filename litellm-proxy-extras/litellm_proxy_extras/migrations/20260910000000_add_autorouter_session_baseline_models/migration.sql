ALTER TABLE "LiteLLM_AutoRouterSession" ADD COLUMN IF NOT EXISTS "baseline_models" JSONB NOT NULL DEFAULT '{}';
