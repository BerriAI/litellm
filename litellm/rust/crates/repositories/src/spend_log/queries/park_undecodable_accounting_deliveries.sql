UPDATE "LiteLLM_SpendLogDelivery"
SET state = 'parked', attempts = attempts + 1, last_error = 'UndecodablePayload'
WHERE delivery_id = ANY($1::text[]) AND state = 'pending'
