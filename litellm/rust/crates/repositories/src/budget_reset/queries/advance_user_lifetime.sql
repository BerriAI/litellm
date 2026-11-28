UPDATE "LiteLLM_UserTable"
SET spend = GREATEST(COALESCE(spend, 0) - $3::double precision, 0),
    budget_reset_at = $2,
    rollover_credit = CASE WHEN $6::double precision IS NULL THEN rollover_credit ELSE $6 END,
    updated_at = NOW() AT TIME ZONE 'UTC'
WHERE user_id = $1 AND budget_reset_at IS NOT DISTINCT FROM $4 AND budget_duration = $5
