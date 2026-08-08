-- Persist the hashed creating API key on managed batch/fine-tune objects so
-- CheckBatchCost can attribute completed-batch spend to the key (key spend /
-- max_budget), not only to user_id / team_id.
ALTER TABLE "LiteLLM_ManagedObjectTable" ADD COLUMN IF NOT EXISTS "created_by_api_key" TEXT;
