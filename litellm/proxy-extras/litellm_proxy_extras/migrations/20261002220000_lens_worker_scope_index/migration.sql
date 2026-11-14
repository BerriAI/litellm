CREATE INDEX IF NOT EXISTS "LiteLLM_LensWorker_active_scope_idx"
ON "LiteLLM_LensWorker" USING GIN ((data->'scope') jsonb_path_ops)
WHERE data @> '{"revoked": false}'::jsonb;
