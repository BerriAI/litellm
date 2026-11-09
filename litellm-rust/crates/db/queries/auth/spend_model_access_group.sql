SELECT COALESCE(spend, 0.0)::float8 AS "spend!" FROM "LiteLLM_ModelAccessGroupBudgetTable" WHERE access_group_name=$1
