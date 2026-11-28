SELECT delivery_id, attempts, accounting_payload
FROM "LiteLLM_SpendLogDelivery"
WHERE state = 'pending'
  AND next_attempt_at <= (NOW() AT TIME ZONE 'UTC')
  AND (attempts > 0) = $2::boolean
ORDER BY next_attempt_at, delivery_id
LIMIT $1::integer
FOR UPDATE SKIP LOCKED
