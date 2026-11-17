CREATE INDEX CONCURRENTLY IF NOT EXISTS "LiteLLM_Lens_jobs_idx"
ON "LiteLLM_Lens" USING GIN ((data->'jobs') jsonb_path_ops);
