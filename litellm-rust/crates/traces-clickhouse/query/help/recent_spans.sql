SELECT
    trace_id, span_id, model, input_tokens, output_tokens, duration_ms
FROM spans
WHERE start_time >= now() - INTERVAL 1 DAY
    AND observation_type = 'llm'
ORDER BY start_time DESC, trace_ref, span_id
LIMIT 100
