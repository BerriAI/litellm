INSERT INTO "LiteLLM_SpendLogDelivery" (delivery_id, request_id, accounting_payload, is_batch, next_attempt_at, created_at)
SELECT delivery_id, request_id, accounting_payload, is_batch, NOW() AT TIME ZONE 'UTC', NOW() AT TIME ZONE 'UTC' FROM jsonb_to_recordset($1::jsonb)
AS inputs(delivery_id text, request_id text, accounting_payload jsonb, is_batch boolean)
ON CONFLICT (delivery_id) DO NOTHING RETURNING delivery_id
