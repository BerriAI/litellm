  AND EngineReceivedMs <= {as_of_ms:UInt64}
ORDER BY Timestamp, EngineReceivedMs, StatusMessage
LIMIT 1 BY TeamId, ApiKeyHash, TraceId, SpanId
)
WHERE (team_id, api_key_hash, trace_id, span_id) > ({after_team:String}, {after_key:String}, {after_trace:String}, {after_span:String})
ORDER BY team_id, api_key_hash, trace_id, span_id
LIMIT {limit:UInt32}
