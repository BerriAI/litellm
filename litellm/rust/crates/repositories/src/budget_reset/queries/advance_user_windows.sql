UPDATE "LiteLLM_UserTable"
SET budget_limits = $3::jsonb,
    updated_at = NOW() AT TIME ZONE 'UTC'
WHERE user_id = $1 AND budget_limits::jsonb = $2::text::jsonb
