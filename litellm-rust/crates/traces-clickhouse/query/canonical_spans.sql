candidate_runs AS (
    SELECT TeamId, ApiKeyHash, TraceId
    FROM owned_runs
    WHERE ({trace_id:String} != '' AND TraceId = {trace_id:String})
       OR ({trace_ref:String} != '' AND hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))) = {trace_ref:String})
    GROUP BY TeamId, ApiKeyHash, TraceId
    UNION ALL
    SELECT TeamId, ApiKeyHash, TraceId
    FROM owned_spans
    WHERE {trace_id:String} = '' AND {trace_ref:String} = ''
      AND EngineReceivedMs <= {as_of_ms:UInt64}
      AND Timestamp >= fromUnixTimestamp64Milli({start_ms:Int64})
      AND Timestamp < fromUnixTimestamp64Milli({end_ms:Int64})
    GROUP BY TeamId, ApiKeyHash, TraceId
),
canonical_spans AS (
    SELECT * FROM owned_spans
    WHERE EngineReceivedMs <= {as_of_ms:UInt64}
      AND Timestamp >= (SELECT min(StartTs) FROM owned_runs WHERE (TeamId, ApiKeyHash, TraceId) IN (SELECT TeamId, ApiKeyHash, TraceId FROM candidate_runs))
      AND (TeamId, ApiKeyHash, TraceId) IN (SELECT TeamId, ApiKeyHash, TraceId FROM candidate_runs)
      AND (TeamId, ApiKeyHash, TraceId) IN (SELECT TeamId, ApiKeyHash, TraceId FROM owned_runs GROUP BY TeamId, ApiKeyHash, TraceId)
    ORDER BY Timestamp, EngineReceivedMs, StatusMessage, Duration, StatusCode
    LIMIT 1 BY TeamId, ApiKeyHash, TraceId, SpanId
)
