UPDATE "LiteLLM_BudgetWindowSpend" AS stored
SET window_start = advanced.window_start,
    spend = 0,
    updated_at = NOW() AT TIME ZONE 'UTC'
FROM unnest($3::text[], $4::timestamp[]) AS advanced(duration, window_start)
WHERE stored.entity_type = $1 AND stored.entity_id = $2
  AND stored.window_duration = advanced.duration
  AND stored.window_start < advanced.window_start
