SELECT app_id,
       customer_org_id,
       spend,
       spend_reset_window,
       tier_budget_id,
       budget_override_id,
       customer_member_budget_id
FROM "LiteLLM_AppCustomerOrgTable"
WHERE app_id = $1 AND customer_org_id = $2
