SELECT spend
FROM "LiteLLM_BudgetWindowSpend"
WHERE entity_type = $1
  AND entity_id = $2
  AND window_duration = $3
  AND window_start >= $4
