INSERT INTO "LiteLLM_BudgetWindowSpend" AS stored
    (entity_type, entity_id, window_duration, window_start, spend, created_at, updated_at)
VALUES ('model_max_budget', $1, $2, $3, 0, $3, $3)
ON CONFLICT (entity_type, entity_id, window_duration) DO UPDATE
SET window_start = EXCLUDED.window_start, spend = 0, updated_at = EXCLUDED.updated_at
WHERE stored.window_start <= $4 AND stored.window_start < EXCLUDED.window_start
