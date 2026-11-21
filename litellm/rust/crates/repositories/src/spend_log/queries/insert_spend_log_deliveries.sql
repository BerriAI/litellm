INSERT INTO "LiteLLM_SpendLogDelivery" (delivery_id, request_id, accounting_payload, is_batch)
SELECT delivery_id, request_id, accounting_payload, is_batch FROM jsonb_to_recordset($1::jsonb)
AS inputs(delivery_id text, request_id text, accounting_payload jsonb, is_batch boolean)
ON CONFLICT (delivery_id) DO NOTHING RETURNING delivery_id
