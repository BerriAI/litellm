SELECT app_id,
       customer_member_id,
       spend,
       spend_reset_window,
       budget_override_id,
       budget_id,
       budget_override_expires_at
FROM "LiteLLM_AppCustomerMemberTable"
WHERE app_id = $1 AND customer_member_id = $2
