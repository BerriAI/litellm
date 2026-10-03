SELECT
    request_id, response_id, trace_id, span_id, model, spend,
    prompt_tokens, completion_tokens, status,
    JSONExtractBool(metadata, 'synthetic_spend') AS synthetic_spend
FROM spend_logs FINAL
WHERE start_time >= now() - INTERVAL 1 DAY
ORDER BY start_time DESC, request_id
LIMIT 100
