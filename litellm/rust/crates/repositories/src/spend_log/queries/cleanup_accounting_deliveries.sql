DELETE FROM "LiteLLM_SpendLogDelivery"
WHERE delivery_id IN (
    SELECT delivery_id
    FROM "LiteLLM_SpendLogDelivery"
    WHERE state = 'applied' AND NOT is_batch
      AND applied_at < (NOW() AT TIME ZONE 'UTC') - make_interval(days => $1::integer)
    ORDER BY applied_at, delivery_id
    LIMIT $2::integer
    FOR UPDATE SKIP LOCKED
)
