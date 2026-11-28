SELECT user_id AS id, budget_duration AS "duration!", budget_reset_at AS reset_at,
       COALESCE(spend, 0.0) AS "spend!", max_budget,
       rollover_max_budget AS rollover_max_budget,
       rollover_credit AS rollover_credit
FROM "LiteLLM_UserTable"
WHERE user_id > $1 AND budget_duration IS NOT NULL
  AND (budget_reset_at < $2 OR budget_reset_at IS NULL)
ORDER BY user_id
LIMIT $3
