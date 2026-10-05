SELECT t.trace_id AS trace_id, t.span_id AS span_id,
       s.request_id AS request_id, s.spend AS spend, s.metadata AS metadata
FROM spans AS t
INNER JOIN (
    SELECT *
    FROM calls
    WHERE start_time >= now() - INTERVAL 1 DAY
) AS s
    ON t.request_id = s.response_id
    AND t.team_id = s.team_id
    AND ((t.user_id != '' AND t.user_id = s.user_id)
        OR (t.api_key_hash != '' AND t.api_key_hash = s.api_key_hash))
WHERE t.start_time >= now() - INTERVAL 1 DAY
    AND t.request_id != ''
LIMIT 100
