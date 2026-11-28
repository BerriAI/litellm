WITH failed AS (
    SELECT delivery.delivery_id
    FROM "LiteLLM_SpendLogDelivery" AS delivery
    JOIN jsonb_to_recordset($1::jsonb) AS claimed(delivery_id text, attempts integer)
      ON claimed.delivery_id = delivery.delivery_id AND claimed.attempts = delivery.attempts
    WHERE delivery.state = 'pending'
    ORDER BY delivery.delivery_id
    FOR UPDATE OF delivery
)
UPDATE "LiteLLM_SpendLogDelivery" AS delivery
SET attempts = attempts + 1, last_error = $2,
    next_attempt_at = (NOW() AT TIME ZONE 'UTC') + make_interval(
        secs => LEAST(300, power(2, LEAST(attempts, 8)) * (0.5 + random()))
    ),
    state = CASE WHEN $3::boolean THEN 'parked' ELSE 'pending' END::"AccountingDeliveryState"
FROM failed
WHERE delivery.delivery_id = failed.delivery_id
