CREATE INDEX CONCURRENTLY IF NOT EXISTS "LiteLLM_LensRun_completed_executions_idx"
ON "LiteLLM_LensRun" USING GIN ((data->'sample'->'executions') jsonb_path_ops)
WHERE data->>'status'='completed';
