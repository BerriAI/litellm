SELECT CASE
    WHEN EXISTS (
        SELECT 1 FROM "LiteLLM_BudgetWindowSpend"
        WHERE entity_type = $1 AND entity_id = $2 AND window_duration = $3
    ) OR $1 NOT IN ('key', 'team', 'user') THEN 0.0::double precision
    ELSE (
        SELECT greatest(
            coalesce(sum(spend), 0.0) - $6,
            coalesce(sum(spend) FILTER (
                WHERE $5::timestamp IS NULL OR "startTime" < date_trunc('second', $5::timestamp)
            ), 0.0)
        )
        FROM "LiteLLM_SpendLogs"
        WHERE "startTime" >= $4
          AND (($1 = 'key' AND api_key = $2) OR ($1 = 'team' AND team_id = $2)
            OR ($1 = 'user' AND "user" = $2))
    )
END AS "base!"
