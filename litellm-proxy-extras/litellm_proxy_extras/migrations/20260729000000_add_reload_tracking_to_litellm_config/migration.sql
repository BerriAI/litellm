-- AlterTable
ALTER TABLE "LiteLLM_Config" ADD COLUMN IF NOT EXISTS "last_run_at" TIMESTAMP(3),
ADD COLUMN IF NOT EXISTS "reload_requested_at" TIMESTAMP(3);

-- Carry over a manual reload that was still pending under the older boolean flag, so
-- pods that have not picked it up yet still fan out after this deploy.
UPDATE "LiteLLM_Config" SET "reload_requested_at" = CURRENT_TIMESTAMP WHERE "param_value"->'force_reload' = 'true'::jsonb;
