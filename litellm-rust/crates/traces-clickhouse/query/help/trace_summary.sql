SELECT id, team_id AS team, api_key_hash AS api_key, trace_id, name,
       span_count AS spans, llm_span_count, error_count AS errors,
       span_input_tokens, span_output_tokens
FROM traces
WHERE start_time >= now() - INTERVAL 1 DAY
ORDER BY team, api_key, trace_id
LIMIT 100
