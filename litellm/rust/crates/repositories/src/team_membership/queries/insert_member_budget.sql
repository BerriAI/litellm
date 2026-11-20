INSERT INTO "LiteLLM_BudgetTable" (
    budget_id,
    allowed_models,
    budget_duration,
    budget_reset_at,
    created_by,
    updated_by,
    created_at,
    updated_at
)
SELECT $1::text,
       $2::text[],
       $3::text,
       $4::timestamp,
       $5::text,
       $6::text,
       NOW() AT TIME ZONE 'UTC',
       NOW() AT TIME ZONE 'UTC'
WHERE NOT EXISTS (
    SELECT 1
    FROM "LiteLLM_TeamMembership"
    WHERE user_id = $7
      AND team_id = $8
)
