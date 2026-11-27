INSERT INTO "LiteLLM_BudgetTable" (
    budget_id,
    max_budget,
    allowed_models,
    created_by,
    updated_by,
    created_at,
    updated_at
)
SELECT $1::text,
       $2::float8,
       COALESCE($3::text[], '{}'),
       $4::text,
       $5::text,
       NOW() AT TIME ZONE 'UTC',
       NOW() AT TIME ZONE 'UTC'
WHERE NOT EXISTS (
    SELECT 1
    FROM "LiteLLM_TeamMembership"
    WHERE user_id = $6
      AND team_id = $7
)
