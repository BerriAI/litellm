DELETE FROM "LiteLLM_SpendLogDelivery" WHERE delivery_id = ANY($1::text[])
