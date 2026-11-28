SELECT window_start, spend FROM "LiteLLM_BudgetWindowSpend"
WHERE entity_type = 'model_max_budget' AND entity_id = $1 AND window_duration = $2
