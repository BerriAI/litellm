SELECT budget_id AS id, budget_duration AS "duration!", budget_reset_at AS reset_at,
       0.0::double precision AS "spend!", max_budget,
       rollover_max_budget AS rollover_max_budget,
       NULL::double precision AS rollover_credit
FROM "LiteLLM_BudgetTable"
WHERE budget_id > $1 AND budget_duration IS NOT NULL
  AND (budget_reset_at < $2 OR budget_reset_at IS NULL)
ORDER BY budget_id
LIMIT $3
