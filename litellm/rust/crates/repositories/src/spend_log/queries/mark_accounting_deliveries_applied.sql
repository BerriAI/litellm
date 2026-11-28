UPDATE "LiteLLM_SpendLogDelivery"
SET state = 'applied', applied_at = (NOW() AT TIME ZONE 'UTC'), last_error = NULL
WHERE delivery_id = ANY($1::text[]) AND state = 'pending'
