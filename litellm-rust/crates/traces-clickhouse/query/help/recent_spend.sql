SELECT
    request_id, response_id, trace_id, span_id, model, spend,
    input_tokens, output_tokens, status
FROM calls
WHERE start_time >= now() - INTERVAL 1 DAY
ORDER BY start_time DESC, request_id
LIMIT 100
