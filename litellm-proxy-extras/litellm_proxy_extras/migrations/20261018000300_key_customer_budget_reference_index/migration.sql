CREATE INDEX CONCURRENTLY IF NOT EXISTS "LiteLLM_VerificationToken_end_user_budget_id_idx" ON "LiteLLM_VerificationToken" USING GIN (("metadata"->'end_user_budget_id') jsonb_path_ops);
