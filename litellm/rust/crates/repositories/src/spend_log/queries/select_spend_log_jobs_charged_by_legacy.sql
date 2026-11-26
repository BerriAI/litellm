SELECT DISTINCT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = ANY($1::text[])
AND status = 'success' AND spend > 0
