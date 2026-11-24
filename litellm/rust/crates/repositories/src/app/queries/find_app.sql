SELECT app_id,
       app_group_id,
       resource_type,
       resource_id,
       alias,
       spend,
       spend_reset_window,
       budget_id,
       default_tier_budget_id,
       default_customer_member_budget_id,
       auto_create_customers,
       customer_margin,
       created_at,
       created_by,
       updated_at,
       updated_by
FROM "LiteLLM_AppTable"
WHERE app_id = $1
