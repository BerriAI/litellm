SELECT jsonb_build_object('window_start', window_start, 'spend', spend) AS "data!: AuthData" FROM "LiteLLM_BudgetWindowSpend" WHERE entity_type=$1 AND entity_id=$2 AND window_duration=$3
