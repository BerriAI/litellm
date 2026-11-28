INSERT INTO "LiteLLM_BudgetWindowSpend" AS stored
    (entity_type, entity_id, window_duration, window_start, spend, created_at, updated_at)
VALUES ($1, $2, $3, $4, $5, (NOW() AT TIME ZONE 'UTC'), (NOW() AT TIME ZONE 'UTC'))
ON CONFLICT (entity_type, entity_id, window_duration) DO UPDATE
SET spend = CASE WHEN stored.window_start >= EXCLUDED.window_start
               THEN stored.spend + $6 ELSE EXCLUDED.spend END,
    window_start = greatest(stored.window_start, EXCLUDED.window_start),
    updated_at = EXCLUDED.updated_at
