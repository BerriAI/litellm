UPDATE "LiteLLM_TeamTable"
SET spend = GREATEST(COALESCE(spend, 0) - $3::double precision, 0),
    budget_reset_at = $2,
    updated_at = NOW() AT TIME ZONE 'UTC'
WHERE team_id = $1 AND budget_reset_at IS NOT DISTINCT FROM $4 AND budget_duration = $5
